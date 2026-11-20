// woof.da.var: the 3D-Var control-variable transform U = U_p U_v U_h, its
// adjoint, and the fixed-order inner products (docs/dev/WRFDA-3DVAR-PLAN.md,
// sections 4.5 and 4.7).
//
// Every kernel follows the statement order of the NumPy reference
// (woof/da/var/reference.py), which follows WRFDA 4.7.1's own routines
// statement by statement and reproduces the serial gfortran build word for
// word (gate O2).  Each binary64 or binary32 operation is spelled as its
// correctly rounded intrinsic (__dadd_rn, __fmul_rn, ...), so no compiler
// option can contract a multiply and an add into one rounding: the words do
// not depend on --fmad or on the numerics mode, and the Metal translation
// runs the same operations.
//
// One source, two real types: VAR_F64 selects binary64 (the oracle tier),
// otherwise binary32 (the production tier).  The inner products accumulate
// in binary64 in both.
//
// Layout (C order, 0-based): a 3-D field is (nx, ny, nz), element
// (i, j, k) at (i * ny + j) * nz + k; evec (ny, nke, mze); val (ny, mzv);
// reg psi_t (ny, nz, nz), psi_ps and psi_chi (ny, nz); coefx, coefy and
// 2-D fields (nx, ny).
//
// Derived from WRFDA 4.7.1 (var/da/da_recursive_filter, da_vtox_transforms,
// da_dynamics), public domain; see licenses/LICENSE-WRF-public-domain.txt.

#ifdef VAR_F64
typedef double real;
#define R_ADD(a, b) __dadd_rn((a), (b))
#define R_SUB(a, b) __dsub_rn((a), (b))
#define R_MUL(a, b) __dmul_rn((a), (b))
#define R_DIV(a, b) __ddiv_rn((a), (b))
#define R_SQRT(a) __dsqrt_rn(a)
#define R_ZERO 0.0
#define R_ONE 1.0
#define R_HALF 0.5
#else
typedef float real;
#define R_ADD(a, b) __fadd_rn((a), (b))
#define R_SUB(a, b) __fsub_rn((a), (b))
#define R_MUL(a, b) __fmul_rn((a), (b))
#define R_DIV(a, b) __fdiv_rn((a), (b))
#define R_SQRT(a) __fsqrt_rn(a)
#define R_ZERO 0.0f
#define R_ONE 1.0f
#define R_HALF 0.5f
#endif

#define GRID_LOOP(t, total) \
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < (total); \
         t += (long long)gridDim.x * blockDim.x)

// ---------------------------------------------------------------- elementwise

extern "C" __global__ void var_sqrt(const real* __restrict__ a, real* __restrict__ out, const long long n) {
    GRID_LOOP(t, n) { out[t] = R_SQRT(a[t]); }
}

// f(i,j,m) = f / px(i,j) (op 0) or f * px(i,j) (op 1)
extern "C" __global__ void var_scale_2d(real* __restrict__ f, const real* __restrict__ px,
                                        const int nx, const int ny, const int mz, const int op) {
    const long long total = (long long)nx * ny * mz;
    GRID_LOOP(t, total) {
        const real p = px[t / mz];
        f[t] = op == 0 ? R_DIV(f[t], p) : R_MUL(f[t], p);
    }
}

// out = out + a  (out zeroed by the caller where WRFDA adds into a zeroed array)
extern "C" __global__ void var_add(real* __restrict__ out, const real* __restrict__ a, const long long n) {
    GRID_LOOP(t, n) { out[t] = R_ADD(out[t], a[t]); }
}

// out = out + a * b
extern "C" __global__ void var_add_prod(real* __restrict__ out, const real* __restrict__ a,
                                        const real* __restrict__ b, const long long n) {
    GRID_LOOP(t, n) { out[t] = R_ADD(out[t], R_MUL(a[t], b[t])); }
}

// out = a * b
extern "C" __global__ void var_mul(real* __restrict__ out, const real* __restrict__ a,
                                   const real* __restrict__ b, const long long n) {
    GRID_LOOP(t, n) { out[t] = R_MUL(a[t], b[t]); }
}

// out(i,j,k) = out + r(j,k) * x(i,j,k)   (x2d: x(i,j))
extern "C" __global__ void var_add_reg_prod(real* __restrict__ out, const real* __restrict__ r,
                                            const real* __restrict__ x, const int nx, const int ny,
                                            const int nz, const int x2d) {
    const long long total = (long long)nx * ny * nz;
    GRID_LOOP(t, total) {
        const int k = (int)(t % nz);
        const long long ij = t / nz;
        const int j = (int)(ij % ny);
        const real xv = x2d ? x[ij] : x[t];
        out[t] = R_ADD(out[t], R_MUL(r[(long long)j * nz + k], xv));
    }
}

// out = a + y  (axpy: y = y + a x)
extern "C" __global__ void var_axpy(real* __restrict__ y, const real* __restrict__ x, const real a,
                                    const long long n) {
    GRID_LOOP(t, n) { y[t] = R_ADD(y[t], R_MUL(a, x[t])); }
}

// out = x / a  (da_minimise_cg: qhat = ghat / sqrt(rrmold), a division, not
// a product with the reciprocal)
extern "C" __global__ void var_div_scalar(real* __restrict__ out, const real* __restrict__ x, const real a,
                                          const long long n) {
    GRID_LOOP(t, n) { out[t] = R_DIV(x[t], a); }
}

// p = -t + b p  (da_minimise_cg: phat = -precon * ghat + ratio * phat, with
// t = precon * ghat; da_calculate_gradj: grad = -grad_jo + jb_factor * cv)
extern "C" __global__ void var_neg_axpby(real* __restrict__ p, const real* __restrict__ t, const real b,
                                         const long long n) {
    GRID_LOOP(i, n) { p[i] = R_ADD(-t[i], R_MUL(b, p[i])); }
}

// ----------------------------------------------------- recursive filter lines
//
// da_recursive_filter_1d / _adj, in place along one line of n values at
// stride s.  The forward pass overwrites a with b then b with c; the adjoint
// overwrites c with b then b with a, each slot read before it is written, in
// exactly the reference's statement order.

__device__ __forceinline__ void rf_forward(real* f, const long long s, const int n, const int pass,
                                           const real alpha) {
    const real one_alpha = R_SUB(R_ONE, alpha);
    const real p3 = R_MUL(R_MUL(alpha, alpha), alpha);
    const real den = R_SUB(R_ONE, R_MUL(alpha, alpha));
    real b0;
    if (pass == 1) {
        b0 = R_MUL(one_alpha, f[0]);
    } else if (pass == 2) {
        b0 = R_DIV(f[0], R_ADD(R_ONE, alpha));
    } else {
        b0 = R_DIV(R_MUL(one_alpha, R_SUB(f[0], R_MUL(p3, f[s]))), R_MUL(den, den));
    }
    f[0] = b0;
    real prev = b0;
    for (int j = 1; j < n; ++j) {
        const real b = R_ADD(R_MUL(alpha, prev), R_MUL(one_alpha, f[(long long)j * s]));
        f[(long long)j * s] = b;
        prev = b;
    }
    const long long last = (long long)(n - 1) * s;
    real c;
    if (pass == 1) {
        c = R_DIV(f[last], R_ADD(R_ONE, alpha));
    } else {
        c = R_DIV(R_MUL(one_alpha, R_SUB(f[last], R_MUL(p3, f[last - s]))), R_MUL(den, den));
    }
    f[last] = c;
    for (int j = n - 2; j >= 0; --j) {
        c = R_ADD(R_MUL(alpha, c), R_MUL(one_alpha, f[(long long)j * s]));
        f[(long long)j * s] = c;
    }
}

__device__ __forceinline__ void rf_adjoint(real* f, const long long s, const int n, const int pass,
                                           const real alpha) {
    const real one_alpha = R_SUB(R_ONE, alpha);
    const real p3 = R_MUL(R_MUL(alpha, alpha), alpha);
    const real den = R_SUB(R_ONE, R_MUL(alpha, alpha));
    const real den2 = R_MUL(den, den);
    // c[j+1] += alpha c[j]; b[j] = one_alpha c[j]
    for (int j = 0; j < n - 1; ++j) {
        const long long a0 = (long long)j * s;
        const real cj = f[a0];
        f[a0 + s] = R_ADD(f[a0 + s], R_MUL(alpha, cj));
        f[a0] = R_MUL(one_alpha, cj);
    }
    const long long last = (long long)(n - 1) * s;
    const real cl = f[last];
    if (pass == 1) {
        f[last] = R_ADD(R_ZERO, R_DIV(cl, R_ADD(R_ONE, alpha)));
    } else {
        f[last] = R_ADD(R_ZERO, R_DIV(R_MUL(one_alpha, cl), den2));
        f[last - s] = R_SUB(f[last - s], R_DIV(R_MUL(R_MUL(one_alpha, p3), cl), den2));
    }
    // b[j-1] += alpha b[j]; a[j] = 0 + one_alpha b[j]
    for (int j = n - 1; j >= 1; --j) {
        const long long a0 = (long long)j * s;
        const real bj = f[a0];
        f[a0 - s] = R_ADD(f[a0 - s], R_MUL(alpha, bj));
        f[a0] = R_ADD(R_ZERO, R_MUL(one_alpha, bj));
    }
    const real b0 = f[0];
    if (pass == 1) {
        f[0] = R_ADD(R_ZERO, R_MUL(one_alpha, b0));
    } else if (pass == 2) {
        f[0] = R_ADD(R_ZERO, R_DIV(b0, R_ADD(R_ONE, alpha)));
    } else {
        f[0] = R_ADD(R_ZERO, R_DIV(R_MUL(one_alpha, b0), den2));
        f[s] = R_SUB(f[s], R_DIV(R_MUL(R_MUL(one_alpha, p3), b0), den2));
    }
}

// One thread per line.  axis 0: x lines, one per (j, m), n = nx;
// axis 1: y lines, one per (i, m), n = ny.  Passes 1..npass forward, or
// npass..1 adjoint, as da_transform_through_rf(_adj).
extern "C" __global__ void var_rf_lines(real* __restrict__ f, const real* __restrict__ alpha,
                                        const int nx, const int ny, const int mz, const int axis,
                                        const int npass, const int adjoint) {
    const long long lines = axis == 0 ? (long long)ny * mz : (long long)nx * mz;
    GRID_LOOP(t, lines) {
        const int m = (int)(t % mz);
        const long long other = t / mz;  // j (axis 0) or i (axis 1)
        long long base, s;
        int n;
        if (axis == 0) {
            base = other * mz + m;
            s = (long long)ny * mz;
            n = nx;
        } else {
            base = other * ny * mz + m;
            s = mz;
            n = ny;
        }
        const real a = alpha[m];
        if (adjoint) {
            for (int p = npass; p >= 1; --p) rf_adjoint(f + base, s, n, p, a);
        } else {
            for (int p = 1; p <= npass; ++p) rf_forward(f + base, s, n, p, a);
        }
    }
}

// ------------------------------------------------------------ vertical EOFs

// vp(i,j,k) = sum_m (evec(j,k,m) * val(j,m)) * vv(i,j,m), m ascending from 0
extern "C" __global__ void var_vvtovp(const real* __restrict__ evec, const real* __restrict__ val,
                                      const real* __restrict__ vv, real* __restrict__ vp,
                                      const int nx, const int ny, const int levels, const int mz,
                                      const int nke, const int mze, const int mzv) {
    const long long total = (long long)nx * ny * levels;
    GRID_LOOP(t, total) {
        const int k = (int)(t % levels);
        const long long ij = t / levels;
        const int j = (int)(ij % ny);
        const real* e = evec + ((long long)j * nke + k) * mze;
        const real* v = val + (long long)j * mzv;
        const real* w = vv + ij * mz;
        real acc = R_ZERO;
        for (int m = 0; m < mz; ++m) acc = R_ADD(acc, R_MUL(R_MUL(e[m], v[m]), w[m]));
        vp[t] = acc;
    }
}

// vv(i,j,m) = sum_k (evec(j,k,m) * val(j,m)) * vp(i,j,k), k ascending from 0
extern "C" __global__ void var_vvtovp_adj(const real* __restrict__ evec, const real* __restrict__ val,
                                          const real* __restrict__ vp, real* __restrict__ vv,
                                          const int nx, const int ny, const int levels, const int mz,
                                          const int nke, const int mze, const int mzv) {
    const long long total = (long long)nx * ny * mz;
    GRID_LOOP(t, total) {
        const int m = (int)(t % mz);
        const long long ij = t / mz;
        const int j = (int)(ij % ny);
        const real vm = val[(long long)j * mzv + m];
        const real* p = vp + ij * levels;
        real acc = R_ZERO;
        for (int k = 0; k < levels; ++k) {
            const real e = evec[((long long)j * nke + k) * mze + m];
            acc = R_ADD(acc, R_MUL(R_MUL(e, vm), p[k]));
        }
        vv[t] = acc;
    }
}

// ----------------------------------------------------------------- balance

// v3(i,j,k) += reg_psi_t(j,k,k1) * v1(i,j,k1), k1 ascending (forward), or
// v1(i,j,k1) += reg_psi_t(j,k,k1) * t(i,j,k), k ascending (adjoint)
extern "C" __global__ void var_reg_psi_t(real* __restrict__ out, const real* __restrict__ rt,
                                         const real* __restrict__ x, const int nx, const int ny,
                                         const int nz, const int adjoint) {
    const long long total = (long long)nx * ny * nz;
    GRID_LOOP(t, total) {
        const int k = (int)(t % nz);
        const long long ij = t / nz;
        const int j = (int)(ij % ny);
        const real* xc = x + ij * nz;
        real acc = out[t];
        if (adjoint) {
            for (int kk = 0; kk < nz; ++kk)
                acc = R_ADD(acc, R_MUL(rt[((long long)j * nz + kk) * nz + k], xc[kk]));
        } else {
            const real* r = rt + ((long long)j * nz + k) * nz;
            for (int k1 = 0; k1 < nz; ++k1) acc = R_ADD(acc, R_MUL(r[k1], xc[k1]));
        }
        out[t] = acc;
    }
}

// v5(i,j) += reg_psi_ps(j,k) * v1(i,j,k), k ascending
extern "C" __global__ void var_reg_psi_ps(real* __restrict__ v5, const real* __restrict__ rp,
                                          const real* __restrict__ v1, const int nx, const int ny,
                                          const int nz) {
    const long long total = (long long)nx * ny;
    GRID_LOOP(t, total) {
        const int j = (int)(t % ny);
        const real* r = rp + (long long)j * nz;
        const real* x = v1 + t * nz;
        real acc = v5[t];
        for (int k = 0; k < nz; ++k) acc = R_ADD(acc, R_MUL(r[k], x[k]));
        v5[t] = acc;
    }
}

// ------------------------------------------------------------- psi/chi to u/v

__device__ __forceinline__ void diff_pair(const int i, const int n, int* ip, int* im) {
    if (i == 0) {
        *ip = 2; *im = 0;
    } else if (i == n - 1) {
        *ip = n - 1; *im = n - 3;
    } else {
        *ip = i + 1; *im = i - 1;
    }
}

// da_psichi_to_uv (regional, A grid), every point except the four corners
extern "C" __global__ void var_psichi_to_uv(const real* __restrict__ psi, const real* __restrict__ chi,
                                            const real* __restrict__ cx, const real* __restrict__ cy,
                                            real* __restrict__ u, real* __restrict__ v,
                                            const int nx, const int ny, const int nz) {
    const long long total = (long long)nx * ny * nz;
    GRID_LOOP(t, total) {
        const int k = (int)(t % nz);
        const long long ij = t / nz;
        const int j = (int)(ij % ny);
        const int i = (int)(ij / ny);
        const bool ei = i == 0 || i == nx - 1, ej = j == 0 || j == ny - 1;
        if (ei && ej) continue;
        int ip, im, jp, jm;
        diff_pair(i, nx, &ip, &im);
        diff_pair(j, ny, &jp, &jm);
        const long long xp = ((long long)ip * ny + j) * nz + k, xm = ((long long)im * ny + j) * nz + k;
        const long long yp = ((long long)i * ny + jp) * nz + k, ym = ((long long)i * ny + jm) * nz + k;
        const real a = cx[ij], b = cy[ij];
        u[t] = R_ADD(-R_MUL(b, R_SUB(psi[yp], psi[ym])), R_MUL(a, R_SUB(chi[xp], chi[xm])));
        v[t] = R_ADD(R_MUL(a, R_SUB(psi[xp], psi[xm])), R_MUL(b, R_SUB(chi[yp], chi[ym])));
    }
}

// the corners: a(corner) = 0.5 (a(x neighbour) + a(y neighbour)); one thread per k
extern "C" __global__ void var_uv_corners(real* __restrict__ u, real* __restrict__ v,
                                          const int nx, const int ny, const int nz) {
    GRID_LOOP(k, (long long)nz) {
        real* fs[2] = {u, v};
        for (int q = 0; q < 2; ++q) {
            real* a = fs[q];
#define AT(I, J) a[((long long)(I) * ny + (J)) * nz + k]
            AT(0, 0) = R_MUL(R_HALF, R_ADD(AT(1, 0), AT(0, 1)));
            AT(nx - 1, 0) = R_MUL(R_HALF, R_ADD(AT(nx - 2, 0), AT(nx - 1, 1)));
            AT(0, ny - 1) = R_MUL(R_HALF, R_ADD(AT(1, ny - 1), AT(0, ny - 2)));
            AT(nx - 1, ny - 1) = R_MUL(R_HALF, R_ADD(AT(nx - 2, ny - 1), AT(nx - 1, ny - 2)));
#undef AT
        }
    }
}

// da_psichi_to_uv_adj's corner statements, on the caller's copies of u, v
extern "C" __global__ void var_uv_corners_adj(real* __restrict__ u, real* __restrict__ v,
                                              const int nx, const int ny, const int nz) {
    GRID_LOOP(k, (long long)nz) {
        const int ie = nx - 1, je = ny - 1;
        real* fs[2] = {u, v};
#define AT(I, J) a[((long long)(I) * ny + (J)) * nz + k]
        for (int q = 0; q < 2; ++q) {
            real* a = fs[q];
            AT(1, 0) = R_ADD(AT(1, 0), R_MUL(R_HALF, AT(0, 0)));
            AT(0, 1) = R_ADD(AT(0, 1), R_MUL(R_HALF, AT(0, 0)));
        }
        for (int q = 0; q < 2; ++q) {
            real* a = fs[q];
            AT(ie - 1, 0) = R_ADD(AT(ie - 1, 0), R_MUL(R_HALF, AT(ie, 0)));
            AT(ie, 1) = R_ADD(AT(ie, 1), R_MUL(R_HALF, AT(ie, 0)));
        }
        for (int q = 0; q < 2; ++q) {
            real* a = fs[q];
            AT(1, je) = R_ADD(AT(1, je), R_MUL(R_HALF, AT(0, je)));
            AT(0, je - 1) = R_ADD(AT(0, je - 1), R_MUL(R_HALF, AT(0, je)));
        }
        for (int q = 0; q < 2; ++q) {
            real* a = fs[q];
            AT(ie - 1, je) = R_ADD(AT(ie - 1, je), R_MUL(R_HALF, AT(ie, je)));
            AT(ie, je - 1) = R_ADD(AT(ie, je - 1), R_MUL(R_HALF, AT(ie, je)));
        }
#undef AT
    }
}

// The adjoint's scatter, written as a gather: each target point (i, j)
// walks its contributions in the order WRFDA's serial scatter adds them
// (the host builds that list once per grid shape).  Entry e of row r:
// src[e] the source point (i * ny + j), op[e] bit 0 the coefficient
// (0 cx, 1 cy), bit 1 the field (0 u, 1 v), bit 2 the sign (0 +, 1 -).
extern "C" __global__ void var_gather_adj(real* __restrict__ out, const int* __restrict__ row,
                                          const int* __restrict__ src, const int* __restrict__ op,
                                          const real* __restrict__ cx, const real* __restrict__ cy,
                                          const real* __restrict__ u, const real* __restrict__ v,
                                          const int nxy, const int nz) {
    const long long total = (long long)nxy * nz;
    GRID_LOOP(t, total) {
        const int k = (int)(t % nz);
        const long long r = t / nz;
        real acc = out[t];
        for (int e = row[r]; e < row[r + 1]; ++e) {
            const long long s = src[e];
            const int o = op[e];
            const real c = (o & 1) ? cy[s] : cx[s];
            const real f = (o & 2) ? v[s * nz + k] : u[s * nz + k];
            const real p = R_MUL(c, f);
            acc = (o & 4) ? R_SUB(acc, p) : R_ADD(acc, p);
        }
        out[t] = acc;
    }
}

// ------------------------------------------------------------ xb -> xa (p, rho)

// da_transform_xtoxa (regional WRF, 3D-Var): one thread per column
extern "C" __global__ void var_xtoxa(const real* __restrict__ qa, const real* __restrict__ ta,
                                     const real* __restrict__ psfc, const real* __restrict__ qb,
                                     const real* __restrict__ psac, const real* __restrict__ dnw,
                                     const real* __restrict__ c1h, const real* __restrict__ c2h,
                                     const real* __restrict__ rhob, const real* __restrict__ pb,
                                     const real* __restrict__ tb, real* __restrict__ pa,
                                     real* __restrict__ rhoa, const int nx, const int ny, const int nz) {
    const long long total = (long long)nx * ny;
    GRID_LOOP(t, total) {
        const long long o = t * nz;
        const real ps = psac[t];
        real sdmd = R_ZERO, s1md = R_ZERO;
        for (int k = 0; k < nz; ++k) {
            const real omq = R_SUB(R_ONE, qb[o + k]);
            const real mr_a = R_DIV(qa[o + k], R_MUL(omq, omq));
            const real mr_b = R_DIV(qb[o + k], omq);
            sdmd = R_ADD(sdmd, R_MUL(mr_a, dnw[k]));
            s1md = R_ADD(s1md, R_MUL(R_ADD(R_ONE, mr_b), dnw[k]));
        }
        const real mu = R_DIV(-R_ADD(psfc[t], R_MUL(ps, sdmd)), s1md);
        real p_hi = R_ZERO;
        for (int k = nz - 1; k >= 0; --k) {
            const real omq = R_SUB(R_ONE, qb[o + k]);
            const real mr_a = R_DIV(qa[o + k], R_MUL(omq, omq));
            const real mr_b = R_DIV(qb[o + k], omq);
            const real term = R_ADD(R_MUL(R_MUL(c1h[k], mu), R_ADD(R_ONE, mr_b)),
                                    R_MUL(R_ADD(R_MUL(c1h[k], ps), c2h[k]), mr_a));
            const real pk = R_SUB(p_hi, R_MUL(term, dnw[k]));
            const real p = R_MUL(R_HALF, R_ADD(pk, p_hi));
            pa[o + k] = p;
            rhoa[o + k] = R_MUL(rhob[o + k], R_SUB(R_DIV(p, pb[o + k]), R_DIV(ta[o + k], tb[o + k])));
            p_hi = pk;
        }
    }
}

// da_transform_xtoxa_adj (no tpw / gpsref): one thread per column.  p, t,
// psfc and q accumulate in place; rho is consumed (zeroed); mra is scratch.
extern "C" __global__ void var_xtoxa_adj(real* __restrict__ pa, real* __restrict__ ta,
                                         real* __restrict__ rhoa, real* __restrict__ psfc,
                                         real* __restrict__ qa, real* __restrict__ mra,
                                         const real* __restrict__ qb, const real* __restrict__ psac,
                                         const real* __restrict__ dnw, const real* __restrict__ c1h,
                                         const real* __restrict__ c2h, const real* __restrict__ rhob,
                                         const real* __restrict__ pb, const real* __restrict__ tb,
                                         const int nx, const int ny, const int nz) {
    const long long total = (long long)nx * ny;
    GRID_LOOP(t, total) {
        const long long o = t * nz;
        const real ps = psac[t];
        for (int k = 0; k < nz; ++k) {
            const real temp = R_MUL(rhoa[o + k], rhob[o + k]);
            pa[o + k] = R_ADD(pa[o + k], R_DIV(temp, pb[o + k]));
            ta[o + k] = R_SUB(ta[o + k], R_DIV(temp, tb[o + k]));
            rhoa[o + k] = R_ZERO;
        }
        real mu = R_ZERO, s1md = R_ZERO, carry = R_ZERO;
        for (int k = 0; k < nz; ++k) {
            const real mr_b = R_DIV(qb[o + k], R_SUB(R_ONE, qb[o + k]));
            s1md = R_ADD(s1md, R_MUL(R_ADD(R_ONE, mr_b), dnw[k]));
            const real half = R_MUL(R_HALF, pa[o + k]);
            const real pk = R_ADD(carry, half);
            real next = R_ADD(R_ZERO, half);
            mu = R_SUB(mu, R_MUL(R_MUL(R_MUL(c1h[k], pk), R_ADD(R_ONE, mr_b)), dnw[k]));
            mra[o + k] = -R_MUL(R_MUL(pk, R_ADD(R_MUL(c1h[k], ps), c2h[k])), dnw[k]);
            carry = R_ADD(next, pk);
        }
        psfc[t] = R_SUB(psfc[t], R_DIV(mu, s1md));
        const real sdmd = -R_DIV(R_MUL(mu, ps), s1md);
        for (int k = 0; k < nz; ++k) {
            const real m = R_ADD(mra[o + k], R_MUL(sdmd, dnw[k]));
            const real d = R_SUB(R_ONE, qb[o + k]);
            qa[o + k] = R_ADD(qa[o + k], R_DIV(m, R_MUL(d, d)));
        }
    }
}

// --------------------------------------------------------- control layout

// cv block (Fortran order, (nx, ny, mz)) <-> C-order field; dir 0 cv -> field
extern "C" __global__ void var_cv_block(real* __restrict__ cv, real* __restrict__ field,
                                        const int nx, const int ny, const int mz, const int dir) {
    const long long total = (long long)nx * ny * mz;
    GRID_LOOP(t, total) {
        const int m = (int)(t % mz);
        const long long ij = t / mz;
        const int j = (int)(ij % ny);
        const int i = (int)(ij / ny);
        const long long f = ((long long)m * ny + j) * nx + i;
        if (dir == 0) field[t] = cv[f];
        else cv[f] = field[t];
    }
}

// ------------------------------------------------------------ inner products
//
// Products and sums in binary64 for both real types (a binary32 product is
// exact in binary64).

// wrfda-serial: da_dot's order, one long left-to-right sum, one thread
extern "C" __global__ void var_dot_serial(const real* __restrict__ x, const real* __restrict__ y,
                                          const long long n, double* __restrict__ out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double acc = 0.0;
    for (long long i = 0; i < n; ++i) acc = __dadd_rn(acc, __dmul_rn((double)x[i], (double)y[i]));
    out[0] = acc;
}

#define VAR_DOT_THREADS 256
#define VAR_DOT_PER_THREAD 16
#define VAR_DOT_CHUNK (VAR_DOT_THREADS * VAR_DOT_PER_THREAD)

__device__ __forceinline__ void tree_256(double* sh) {
    for (int s = VAR_DOT_THREADS / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] = __dadd_rn(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
}

// fixed-tree, stage 1: block b sums chunk b (4,096 values): each thread its
// 16 consecutive values in index order, then a fixed binary tree.  Launch
// with exactly VAR_DOT_THREADS threads and one block per chunk.
extern "C" __global__ void var_dot_chunks(const real* __restrict__ x, const real* __restrict__ y,
                                          const long long n, double* __restrict__ partial) {
    __shared__ double sh[VAR_DOT_THREADS];
    const long long start = (long long)blockIdx.x * VAR_DOT_CHUNK + (long long)threadIdx.x * VAR_DOT_PER_THREAD;
    double acc = 0.0;
    for (int q = 0; q < VAR_DOT_PER_THREAD; ++q) {
        const long long i = start + q;
        if (i < n) acc = __dadd_rn(acc, __dmul_rn((double)x[i], (double)y[i]));
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    tree_256(sh);
    if (threadIdx.x == 0) partial[blockIdx.x] = sh[0];
}

// fixed-tree, stage 2: one block; thread t sums partials t, t + 256, ...
// in order, then the same tree.
extern "C" __global__ void var_dot_combine(const double* __restrict__ partial, const long long count,
                                           double* __restrict__ out) {
    __shared__ double sh[VAR_DOT_THREADS];
    double acc = 0.0;
    for (long long q = threadIdx.x; q < count; q += VAR_DOT_THREADS) acc = __dadd_rn(acc, partial[q]);
    sh[threadIdx.x] = acc;
    __syncthreads();
    tree_256(sh);
    if (threadIdx.x == 0) out[0] = sh[0];
}
