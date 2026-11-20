// woof.da.var: WRFDA's linearised observation operators H' (a gather) and
// their adjoints H'^T (a fixed-order segment sum) on the device
// (docs/dev/WRFDA-3DVAR-PLAN.md, sections 4.5 and 4.6; plan item V1.6).
//
// The arithmetic is binary64 for both tiers, each operation spelled as its
// correctly rounded intrinsic, in the order of the host operators of
// woof/da/var/obsop_wrfda.py (which follow WRFDA 4.7.1's
// da_interp_lin_3d/_2d, da_transform_xtogpsref_lin/_adj and
// da_transform_xtotpw_adj statement by statement), so the words are the
// host's on CUDA and on Metal's soft binary64.  The fields arrive in the
// tier's real type (VAR_F64: binary64, otherwise binary32; a binary32 value
// is exact in binary64) and the adjoint's fields leave in it (one
// round-to-nearest per value, as the host's cast).
//
// The adjoint scatter is written as a gather: the host lists, per model
// column, every contribution in the order the serial loops add them
// (types in da_transform_xtoy_adj's order, fields, reports, corners), and
// one thread per (column, level) adds them in that order.  No atomics, so
// the words do not depend on scheduling.
//
// Layout (C order, 0-based): a 3-D field is (nx, ny, nz), element (i, j, k)
// at (i * ny + j) * nz + k; a 2-D field (nx, ny).
//
// Derived from WRFDA 4.7.1 (var/da/da_interpolation, da_transform_xtoy_*,
// da_transform_xtogpsref*, da_transform_xtotpw*), public domain; see
// licenses/LICENSE-WRF-public-domain.txt.

#ifdef VAR_F64
typedef double real;
#define TO_REAL(x) (x)
#else
typedef float real;
#define TO_REAL(x) __double2float_rn(x)
#endif

#define D_ADD(a, b) __dadd_rn((a), (b))
#define D_SUB(a, b) __dsub_rn((a), (b))
#define D_MUL(a, b) __dmul_rn((a), (b))
#define D_DIV(a, b) __ddiv_rn((a), (b))

#define GRID_LOOP(t, total) \
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < (total); \
         t += (long long)gridDim.x * blockDim.x)

// dym * (dxm * f00 + dx * f10) + dy * (dxm * f01 + dx * f11)
__device__ __forceinline__ double obs_hcol(const double f00, const double f10, const double f01,
                                           const double f11, const double dx, const double dy,
                                           const double dxm, const double dym) {
    return D_ADD(D_MUL(dym, D_ADD(D_MUL(dxm, f00), D_MUL(dx, f10))),
                 D_MUL(dy, D_ADD(D_MUL(dxm, f01), D_MUL(dx, f11))));
}

// da_interp_lin_3d at one observation level: dzm * fmz(k0) + dz * fmz(k0 + 1),
// 0 where k = 0 (no level).  Per report: i0, j0 (0-based lower-left corner)
// and the weights; per level: rep, k (1-based), dz, dzm.
template <typename T>
__device__ __forceinline__ double obs_lin3(const T* __restrict__ f, const int ny, const int nz,
                                           const int* __restrict__ i0, const int* __restrict__ j0,
                                           const double* __restrict__ dx, const double* __restrict__ dy,
                                           const double* __restrict__ dxm, const double* __restrict__ dym,
                                           const int* __restrict__ rep, const int* __restrict__ kk,
                                           const double* __restrict__ dz, const double* __restrict__ dzm,
                                           const long long t) {
    const int k = kk[t];
    if (k <= 0) return 0.0;
    const int r = rep[t];
    const long long a = (long long)i0[r] * ny + j0[r];
    const long long c00 = a * nz, c10 = (a + ny) * nz, c01 = (a + 1) * nz, c11 = (a + ny + 1) * nz;
    const int k0 = k - 1;
    const double x = dx[r], yy = dy[r], xm = dxm[r], ym = dym[r];
    const double lo = obs_hcol((double)f[c00 + k0], (double)f[c10 + k0], (double)f[c01 + k0],
                               (double)f[c11 + k0], x, yy, xm, ym);
    const double hi = obs_hcol((double)f[c00 + k0 + 1], (double)f[c10 + k0 + 1], (double)f[c01 + k0 + 1],
                               (double)f[c11 + k0 + 1], x, yy, xm, ym);
    return D_ADD(D_MUL(dzm[t], lo), D_MUL(dz[t], hi));
}

// one thread per observation level, from a field of the tier's real type
extern "C" __global__ void var_obs_tl3(const real* __restrict__ f, const int ny, const int nz,
                                       const int* __restrict__ i0, const int* __restrict__ j0,
                                       const double* __restrict__ dx, const double* __restrict__ dy,
                                       const double* __restrict__ dxm, const double* __restrict__ dym,
                                       const int* __restrict__ rep, const int* __restrict__ kk,
                                       const double* __restrict__ dz, const double* __restrict__ dzm,
                                       const long long nrows, double* __restrict__ y, const long long y0) {
    GRID_LOOP(t, nrows) { y[y0 + t] = obs_lin3<real>(f, ny, nz, i0, j0, dx, dy, dxm, dym, rep, kk, dz, dzm, t); }
}

// the same from a binary64 field (gpsref's refractivity increment)
extern "C" __global__ void var_obs_tl3_f64(const double* __restrict__ f, const int ny, const int nz,
                                           const int* __restrict__ i0, const int* __restrict__ j0,
                                           const double* __restrict__ dx, const double* __restrict__ dy,
                                           const double* __restrict__ dxm, const double* __restrict__ dym,
                                           const int* __restrict__ rep, const int* __restrict__ kk,
                                           const double* __restrict__ dz, const double* __restrict__ dzm,
                                           const long long nrows, double* __restrict__ y, const long long y0) {
    GRID_LOOP(t, nrows) {
        y[y0 + t] = obs_lin3<double>(f, ny, nz, i0, j0, dx, dy, dxm, dym, rep, kk, dz, dzm, t);
    }
}

// da_interp_lin_2d at level 1, one thread per report
extern "C" __global__ void var_obs_tl2(const real* __restrict__ f, const int ny,
                                       const int* __restrict__ i0, const int* __restrict__ j0,
                                       const double* __restrict__ dx, const double* __restrict__ dy,
                                       const double* __restrict__ dxm, const double* __restrict__ dym,
                                       const long long nrep, double* __restrict__ y, const long long y0) {
    GRID_LOOP(r, nrep) {
        const long long a = (long long)i0[r] * ny + j0[r];
        y[y0 + r] = obs_hcol((double)f[a], (double)f[a + ny], (double)f[a + 1], (double)f[a + ny + 1], dx[r], dy[r],
                        dxm[r], dym[r]);
    }
}

// da_interp_lin_3d_adj's per-report profile, one thread per report: fmz is
// zeroed, then each level (in order) adds dzm * y and dz * y at k0, k0 + 1.
// The type's levels start at y[y0]; report r's profile is row r0 + r of fmz.
extern "C" __global__ void var_obs_fmz(const double* __restrict__ y, const long long y0,
                                       const long long* __restrict__ off, const int* __restrict__ kk,
                                       const double* __restrict__ dz, const double* __restrict__ dzm,
                                       const long long nrep, const int kte, double* __restrict__ fmz,
                                       const long long r0) {
    GRID_LOOP(r, nrep) {
        double* row = fmz + (r0 + r) * kte;
        for (int k = 0; k < kte; ++k) row[k] = 0.0;
        for (long long t = off[r]; t < off[r + 1]; ++t) {
            const int k = kk[t];
            if (k <= 0) continue;
            const double v = y[y0 + t];
            row[k - 1] = D_ADD(D_MUL(dzm[t], v), row[k - 1]);
            row[k] = D_ADD(D_MUL(dz[t], v), row[k]);
        }
    }
}

// The adjoint scatter into one 3-D field, as a gather: thread (column, k)
// adds its column's entries row[col]..row[col + 1] in order.  Entry e adds
// c = w[e] * s[src[e] * nz + k] (only at level lev[e] when lev[e] >= 0),
// times bg(col, k) when flag[e] is set; out starts from zero.
extern "C" __global__ void var_obs_gather3(double* __restrict__ out, const int* __restrict__ row,
                                           const int* __restrict__ src, const double* __restrict__ w,
                                           const int* __restrict__ lev, const int* __restrict__ flag,
                                           const double* __restrict__ s, const double* __restrict__ bg,
                                           const long long ncol, const int nz) {
    GRID_LOOP(t, ncol * nz) {
        const int k = (int)(t % nz);
        const long long col = t / nz;
        double acc = 0.0;
        for (int e = row[col]; e < row[col + 1]; ++e) {
            const int l = lev[e];
            if (l >= 0 && l != k) continue;
            double c = D_MUL(w[e], s[(long long)src[e] * nz + k]);
            if (flag[e]) c = D_MUL(c, bg[t]);
            acc = D_ADD(acc, c);
        }
        out[t] = acc;
    }
}

// The same for a 2-D field: entry e adds w[e] * v[src[e]].
extern "C" __global__ void var_obs_gather2(double* __restrict__ out, const int* __restrict__ row,
                                           const int* __restrict__ src, const double* __restrict__ w,
                                           const double* __restrict__ v, const long long ncol) {
    GRID_LOOP(col, ncol) {
        double acc = 0.0;
        for (int e = row[col]; e < row[col + 1]; ++e) acc = D_ADD(acc, D_MUL(w[e], v[src[e]]));
        out[col] = acc;
    }
}

// da_transform_xtogpsref_lin, elementwise, with the background pieces
// precomputed as the host forms them: a = 0.776 pb, b = tb tb, p9 = a / tb,
// c = 0.622 + 0.378 qb, d = tb 0.378, e = tb c, f = coeff qb, g = e e,
// h = 1 + f / e; cf[0] is the refractivity coefficient (a one-value buffer).
extern "C" __global__ void var_obs_gpsref_tl(const real* __restrict__ ta, const real* __restrict__ qa,
                                             const double* __restrict__ a, const double* __restrict__ b,
                                             const double* __restrict__ p9, const double* __restrict__ c,
                                             const double* __restrict__ d, const double* __restrict__ e,
                                             const double* __restrict__ f, const double* __restrict__ g,
                                             const double* __restrict__ h, const double* __restrict__ cf,
                                             const long long n, double* __restrict__ ref) {
    const double coeff = cf[0];
    GRID_LOOP(t, n) {
        const double tv = (double)ta[t], qv = (double)qa[t];
        const double partone = -D_DIV(D_MUL(a[t], tv), b[t]);
        const double dividnd = D_ADD(D_MUL(tv, c[t]), D_MUL(d[t], qv));
        const double parttwo = D_SUB(D_DIV(D_MUL(coeff, qv), e[t]), D_DIV(D_MUL(f[t], dividnd), g[t]));
        ref[t] = D_ADD(D_MUL(p9[t], parttwo), D_MUL(partone, h[t]));
    }
}

// da_transform_xtogpsref_adj into gq, gt (binary64 adjoint fields)
extern "C" __global__ void var_obs_gpsref_adj(const double* __restrict__ ref, double* __restrict__ gq,
                                              double* __restrict__ gt, const double* __restrict__ a,
                                              const double* __restrict__ b, const double* __restrict__ p9,
                                              const double* __restrict__ c, const double* __restrict__ d,
                                              const double* __restrict__ e, const double* __restrict__ f,
                                              const double* __restrict__ g, const double* __restrict__ h,
                                              const double* __restrict__ cf, const long long n) {
    const double coeff = cf[0];
    GRID_LOOP(t, n) {
        const double r = ref[t];
        const double partone = D_MUL(r, h[t]);
        const double parttwo = D_MUL(r, p9[t]);
        double q = D_ADD(gq[t], D_DIV(D_MUL(coeff, parttwo), e[t]));
        const double dividnd = -D_DIV(D_MUL(f[t], parttwo), g[t]);
        double tt = D_ADD(gt[t], D_MUL(dividnd, c[t]));
        q = D_ADD(q, D_MUL(d[t], dividnd));
        tt = D_SUB(tt, D_DIV(D_MUL(a[t], partone), b[t]));
        gq[t] = q;
        gt[t] = tt;
    }
}

// da_transform_xtotpw_adj: pw = 0.1 tpw; per level dzpw = dh(k) pw,
// gq += dzpw rho_b, grho += dzpw q_b.  One thread per column.
extern "C" __global__ void var_obs_tpw_adj(const double* __restrict__ tpw, double* __restrict__ gq,
                                           double* __restrict__ grho, const double* __restrict__ dh,
                                           const double* __restrict__ rhob, const double* __restrict__ qb,
                                           const long long ncol, const int nz) {
    GRID_LOOP(col, ncol) {
        const double pw = D_MUL(0.1, tpw[col]);
        for (int k = 0; k < nz; ++k) {
            const long long t = col * nz + k;
            const double dzpw = D_MUL(dh[t], pw);
            gq[t] = D_ADD(gq[t], D_MUL(dzpw, rhob[t]));
            grho[t] = D_ADD(grho[t], D_MUL(dzpw, qb[t]));
        }
    }
}

// da_calculate_grady on the device for the Hessian-vector product: y (H'x,
// binary64) is rounded once to the real type, set to 0 where mask is 0, and
// g = -y / (err * err), one rounding per operation in the real type, as the
// host's NumPy statements in that type.
extern "C" __global__ void var_obs_grady(const double* __restrict__ y, const int* __restrict__ mask,
                                         const real* __restrict__ err, real* __restrict__ g, const long long n) {
    GRID_LOOP(t, n) {
        const real zero = (real)0;
        const real r = mask[t] ? (real)TO_REAL(y[t]) : zero;
        const real e = err[t];
#ifdef VAR_F64
        g[t] = __ddiv_rn(-r, __dmul_rn(e, e));
#else
        g[t] = __fdiv_rn(-r, __fmul_rn(e, e));
#endif
    }
}

// binary64 -> the tier's real type, one rounding (round to nearest even)
extern "C" __global__ void var_obs_to_real(const double* __restrict__ a, real* __restrict__ out,
                                           const long long n) {
    GRID_LOOP(t, n) { out[t] = TO_REAL(a[t]); }
}
