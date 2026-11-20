// Tangent-linear and adjoint of woof's strict flux-form advection
// (P3a item 6): flux_div_scalar, flux_div_u, flux_div_v and flux_div_w
// (woof/core/kernels/advection.cu, the GPUWM_WRF_EXACT_C_ADVECTION
// paths), periodic, open and specified lateral boundaries, map factors,
// h_*_adv_order 3-6 (ADV_H) and v_*_adv_order 2-6 (vorder).
//
// The oracle is WRFPLUS 4.7.1 G_/A_ADVECT_SCALAR, _U, _V and _W; see
// docs/dev/P3A-WRF-DRYCORE-ADJOINT.md.
//
// wreal is binary64 unless the unit is built with -DWRFPLUS_REAL=float.
// The TL kernels are the strict primal statements on a value/tangent pair
// (Dl): the value half repeats the primal's operations in its order, so
// the float instantiation's base state gives the primal's words.  Plain
// '+', '-', '*' are the primal's __f*_rn only with --fmad=false, which the
// wrapper passes; division by the stencil denominators is __fdiv_rn.
//
// Every advected tendency is ADDED (tend += -div F), so the tendency's
// adjoint passes through unchanged and the AD only adds into the field's
// and the mass fluxes' adjoints.  The AD is in gather form: one thread per
// output word, which collects every face flux that reads it (one writer
// per word, no atomics).  A face flux is linear in the advected field with
// weights V*c + sg*|V|*e (V the face's advecting mass flux), so the field
// adjoint takes the weights and the flux adjoint takes c.q + sg*sign(V)*e.q.
//
// Layout (k, j, i): mass (nz, ny, nx), u (nz, ny, nx+1), v (nz, ny+1, nx),
// w (nz+1, ny, nx); ru, rv, rw as u, v, w; msf at the tendency points.

#ifndef WRFPLUS_REAL
#define WRFPLUS_REAL double
#endif

// WRFPLUS_TL 0 builds the base-only forward (model.step_base): every
// perturbation store of the TL kernels is skipped, so the compiler drops
// the tangent arithmetic and loads; the base words are unchanged.
#ifndef WRFPLUS_TL
#define WRFPLUS_TL 1
#endif
typedef WRFPLUS_REAL wreal;

#ifndef ADV_H
#define ADV_H 5
#endif

#define R3(k, j, i, nyy, nxx) \
    ((size_t)(k) * (size_t)(nyy) * (size_t)(nxx) + (size_t)(j) * (size_t)(nxx) + (size_t)(i))

static __device__ __forceinline__ int awrap(int i, int n)
{
    int r = i < 0 ? i + n : (i >= n ? i - n : i);
    if ((unsigned)r >= (unsigned)n) {          // offsets beyond one period
        r = i % n;
        r = r < 0 ? r + n : r;
    }
    return r;
}

static __device__ __forceinline__ float wdiv(float a, float b) { return __fdiv_rn(a, b); }
static __device__ __forceinline__ double wdiv(double a, double b) { return a / b; }
static __device__ __forceinline__ float wsign(float v) { return copysignf(1.0f, v); }
static __device__ __forceinline__ double wsign(double v) { return copysign(1.0, v); }
static __device__ __forceinline__ float wmin0(float v) { return fminf(v, 0.0f); }
static __device__ __forceinline__ double wmin0(double v) { return fmin(v, 0.0); }
static __device__ __forceinline__ float wmax0(float v) { return fmaxf(v, 0.0f); }
static __device__ __forceinline__ double wmax0(double v) { return fmax(v, 0.0); }

// ------------------------------------------------------------ value/tangent

struct Dl { wreal v; wreal d; };

static __device__ __forceinline__ Dl dl(wreal v, wreal d)
{
    Dl r;
    r.v = v;
    r.d = d;
    return r;
}
static __device__ __forceinline__ Dl operator+(Dl a, Dl b) { return dl(a.v + b.v, a.d + b.d); }
static __device__ __forceinline__ Dl operator-(Dl a, Dl b) { return dl(a.v - b.v, a.d - b.d); }
static __device__ __forceinline__ Dl operator*(Dl a, Dl b)
{
    return dl(a.v * b.v, a.d * b.v + a.v * b.d);
}
static __device__ __forceinline__ Dl operator*(wreal a, Dl b) { return dl(a * b.v, a * b.d); }
static __device__ __forceinline__ Dl ddiv(Dl a, wreal c) { return dl(wdiv(a.v, c), a.d / c); }
static __device__ __forceinline__ Dl dmin0(Dl a) { return dl(wmin0(a.v), a.v < (wreal)0 ? a.d : (wreal)0); }
static __device__ __forceinline__ Dl dmax0(Dl a) { return dl(wmax0(a.v), a.v > (wreal)0 ? a.d : (wreal)0); }
static __device__ __forceinline__ Dl ld(const wreal* a, const wreal* ad, size_t at)
{
    return dl(a[at], ad[at]);
}

// ----------------------------------------------- the primal's face fluxes
// (the GPUWM_WRF_EXACT_C_ADVECTION statements of advection.cu)

static __device__ __forceinline__ Dl tf2(Dl qm1, Dl q0, Dl vel)
{
    return ((wreal)0.5 * vel) * (q0 + qm1);
}
static __device__ __forceinline__ Dl tf4(Dl qm2, Dl qm1, Dl q0, Dl qp1, Dl vel)
{
    return vel * ddiv((wreal)7 * (q0 + qm1) - (qp1 + qm2), (wreal)12);
}
static __device__ __forceinline__ Dl tf6(Dl qm3, Dl qm2, Dl qm1, Dl q0, Dl qp1, Dl qp2, Dl vel)
{
    return vel * ddiv((wreal)37 * (q0 + qm1) - (wreal)8 * (qp1 + qm2) + (qp2 + qm3), (wreal)60);
}
static __device__ __forceinline__ Dl tc3(Dl qm2, Dl qm1, Dl q0, Dl qp1)
{
    return ddiv((wreal)7 * (q0 + qm1) - (qp1 + qm2), (wreal)12);
}
static __device__ __forceinline__ Dl td3(Dl qm2, Dl qm1, Dl q0, Dl qp1, Dl vel)
{
    return ddiv(wsign(vel.v) * ((qp1 - qm2) - (wreal)3 * (q0 - qm1)), (wreal)12);
}
static __device__ __forceinline__ Dl tc5(Dl qm3, Dl qm2, Dl qm1, Dl q0, Dl qp1, Dl qp2)
{
    return ddiv((wreal)37 * (q0 + qm1) - (wreal)8 * (qp1 + qm2) + (qp2 + qm3), (wreal)60);
}
static __device__ __forceinline__ Dl td5(Dl qm3, Dl qm2, Dl qm1, Dl q0, Dl qp1, Dl qp2, Dl vel)
{
    return ddiv(wsign(vel.v) * ((qp2 - qm3) - (wreal)5 * (qp1 - qm2) + (wreal)10 * (q0 - qm1)),
                (wreal)60);
}
// vertical flux3 (Omega-signed dissipation)
static __device__ __forceinline__ Dl tf3v(Dl qm2, Dl qm1, Dl q0, Dl qp1, Dl vel)
{
    return vel * (tc3(qm2, qm1, q0, qp1) - td3(qm2, qm1, q0, qp1, vel));
}
// horizontal flux3 (flux3h)
static __device__ __forceinline__ Dl tf3h(Dl qm2, Dl qm1, Dl q0, Dl qp1, Dl vel)
{
#if ADV_H == 4 || ADV_H == 6
    return vel * ddiv((wreal)7 * (q0 + qm1) - (qp1 + qm2), (wreal)12);
#else
    return vel * (tc3(qm2, qm1, q0, qp1) + td3(qm2, qm1, q0, qp1, vel));
#endif
}
// vertical flux5 (flux5v)
static __device__ __forceinline__ Dl tf5v(Dl qm3, Dl qm2, Dl qm1, Dl q0, Dl qp1, Dl qp2, Dl vel)
{
    return vel * (tc5(qm3, qm2, qm1, q0, qp1, qp2) + td5(qm3, qm2, qm1, q0, qp1, qp2, vel));
}
// horizontal flux5 macro (the h_*_adv_order the unit is built for)
static __device__ __forceinline__ Dl tfh(Dl qm3, Dl qm2, Dl qm1, Dl q0, Dl qp1, Dl qp2, Dl vel)
{
#if ADV_H == 3
    return vel * (tc3(qm2, qm1, q0, qp1) + td3(qm2, qm1, q0, qp1, vel));
#elif ADV_H == 4
    return vel * ddiv((wreal)7 * (q0 + qm1) - (qp1 + qm2), (wreal)12);
#elif ADV_H == 6
    return vel * ddiv((wreal)37 * (q0 + qm1) - (wreal)8 * (qp1 + qm2) + (qp2 + qm3), (wreal)60);
#else
    return vel * (tc5(qm3, qm2, qm1, q0, qp1, qp2) - td5(qm3, qm2, qm1, q0, qp1, qp2, vel));
#endif
}

#define QX(c) ld(q, qd, R3(k, j, (c), nyq, nxs))
// x-face f of a cell-type field (xface_cell_open / xface_cell_per)
static __device__ Dl tx_cell(const wreal* q, const wreal* qd, Dl vel, int k, int j, int f,
                             int n, int nyq, int nxs, int open)
{
    if (open) {
        if (f <= 0 || f >= n) return dl(0, 0);
        int d = min(f, n - f);
        if (d == 1) return tf2(QX(f - 1), QX(f), vel);
        if (d == 2) return tf3h(QX(f - 2), QX(f - 1), QX(f), QX(f + 1), vel);
        return tfh(QX(f - 3), QX(f - 2), QX(f - 1), QX(f), QX(f + 1), QX(f + 2), vel);
    }
    return tfh(QX(awrap(f - 3, n)), QX(awrap(f - 2, n)), QX(awrap(f - 1, n)), QX(awrap(f, n)),
               QX(awrap(f + 1, n)), QX(awrap(f + 2, n)), vel);
}
#undef QX
#define QY(r) ld(q, qd, R3(k, (r), i, nyq, nxs))
// y-face g of a cell-type field (yface_cell_open / yface_cell_per)
static __device__ Dl ty_cell(const wreal* q, const wreal* qd, Dl vel, int k, int g, int i,
                             int n, int nyq, int nxs, int open)
{
    if (open) {
        if (g <= 0 || g >= n) return dl(0, 0);
        int d = min(g, n - g);
        if (d == 1) return tf2(QY(g - 1), QY(g), vel);
        if (d == 2) return tf3h(QY(g - 2), QY(g - 1), QY(g), QY(g + 1), vel);
        return tfh(QY(g - 3), QY(g - 2), QY(g - 1), QY(g), QY(g + 1), QY(g + 2), vel);
    }
    return tfh(QY(awrap(g - 3, n)), QY(awrap(g - 2, n)), QY(awrap(g - 1, n)), QY(awrap(g, n)),
               QY(awrap(g + 1, n)), QY(awrap(g + 2, n)), vel);
}
#undef QY
#define QZ(l) ld(q, qd, R3((l), j, i, nyq, nxs))
// half-level vertical face kf (zface_half)
static __device__ Dl tz_half(const wreal* q, const wreal* qd, Dl vel, int kf, int j, int i,
                             int nz, int nyq, int nxs, const wreal* fnm, const wreal* fnp,
                             int vorder)
{
    if (kf == 0 || kf == nz) return dl(0, 0);
    if (vorder == 2 || kf == 1 || kf == nz - 1)
        return vel * (fnm[kf] * QZ(kf) + fnp[kf] * QZ(kf - 1));
    if (vorder == 6 && kf >= 3 && kf <= nz - 3)
        return tf6(QZ(kf - 3), QZ(kf - 2), QZ(kf - 1), QZ(kf), QZ(kf + 1), QZ(kf + 2), vel);
    if (vorder == 4 || vorder == 6)
        return tf4(QZ(kf - 2), QZ(kf - 1), QZ(kf), QZ(kf + 1), vel);
    if (vorder == 5 && kf >= 3 && kf <= nz - 3)
        return tf5v(QZ(kf - 3), QZ(kf - 2), QZ(kf - 1), QZ(kf), QZ(kf + 1), QZ(kf + 2), vel);
    return tf3v(QZ(kf - 2), QZ(kf - 1), QZ(kf), QZ(kf + 1), vel);
}
// w's mass-level vertical face m (advect_w's vflux ladder)
static __device__ Dl tz_w(const wreal* q, const wreal* qd, Dl vel, int m, int j, int i,
                          int nz, int nyq, int nxs, int vorder)
{
    if (vorder == 2 || vorder == 4 || vorder == 6) {
        if (vorder == 2 || m == 0 || m == nz - 1) return tf2(QZ(m), QZ(m + 1), vel);
        if (vorder == 6 && m >= 2 && m <= nz - 3)
            return tf6(QZ(m - 2), QZ(m - 1), QZ(m), QZ(m + 1), QZ(m + 2), QZ(m + 3), vel);
        return tf4(QZ(m - 1), QZ(m), QZ(m + 1), QZ(m + 2), vel);
    }
    if (m == 0 || m == nz - 1) return tf2(QZ(m), QZ(m + 1), vel);
    if (vorder == 5 && m >= 2 && m <= nz - 3)
        return tf5v(QZ(m - 2), QZ(m - 1), QZ(m), QZ(m + 1), QZ(m + 2), QZ(m + 3), vel);
    return tf3v(QZ(m - 1), QZ(m), QZ(m + 1), QZ(m + 2), vel);
}
#undef QZ

#if ADV_H == 3 || ADV_H == 4
#define ADV_X_FIRST 1
#else
#define ADV_X_FIRST 0
#endif

#define TL_ARGS                                                                   \
    wreal* __restrict__ tend, wreal* __restrict__ tendd,                          \
    const wreal* __restrict__ q, const wreal* __restrict__ qd,                    \
    const wreal* __restrict__ ru, const wreal* __restrict__ rud,                  \
    const wreal* __restrict__ rv, const wreal* __restrict__ rvd,                  \
    const wreal* __restrict__ rw, const wreal* __restrict__ rwd,                  \
    const wreal* __restrict__ msf, const wreal* __restrict__ rdz,                 \
    const wreal* __restrict__ fnm, const wreal* __restrict__ fnp,                 \
    int nl, wreal dx_inv, wreal dy_inv, int nz, int ny, int nx,                   \
    int open_x, int open_y, int spec, int vorder

#define RU(k, j, f) ld(ru, rud, R3((k), (j), (f), ny, nx + 1))
#define RV(k, g, i) ld(rv, rvd, R3((k), (g), (i), ny + 1, nx))
#define RW(k, j, i) ld(rw, rwd, R3((k), (j), (i), ny, nx))

// ------------------------------------------------------------ scalar TL

extern "C" __global__ void adv_scalar_tl(TL_ARGS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    const int nyq = ny, nxs = nx;
    wreal mscale = msf[(size_t)j * nx + i];
    size_t at = R3(k, j, i, ny, nx);
    Dl res;
#define Q(kk, jj, ii) ld(q, qd, R3((kk), (jj), (ii), ny, nx))
    if (open_x || open_y) {
        Dl tx = dl(0, 0), ty = dl(0, 0);
        bool ax = false, ay = false;
        if (open_x && (i == 0 || i == nx - 1)) {
            if (spec) {
            } else if (i == 0) {
                Dl ru0 = RU(k, j, 0), ru1 = RU(k, j, 1);
                Dl ub = dmin0((wreal)0.5 * (ru0 + ru1));
                tx = dx_inv * (ub * (Q(k, j, 1) - Q(k, j, 0)) + Q(k, j, 0) * (ru1 - ru0));
                ax = true;
            } else {
                Dl ru0 = RU(k, j, nx - 1), ru1 = RU(k, j, nx);
                Dl ub = dmax0((wreal)0.5 * (ru0 + ru1));
                tx = dx_inv * (ub * (Q(k, j, nx - 1) - Q(k, j, nx - 2))
                               + Q(k, j, nx - 1) * (ru1 - ru0));
                ax = true;
            }
        } else {
            Dl fx0 = tx_cell(q, qd, RU(k, j, i), k, j, i, nx, nyq, nxs, open_x);
            Dl fx1 = tx_cell(q, qd, RU(k, j, i + 1), k, j, i + 1, nx, nyq, nxs, open_x);
            tx = (mscale * dx_inv) * (fx1 - fx0);
            ax = true;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
            } else if (j == 0) {
                Dl rv0 = RV(k, 0, i), rv1 = RV(k, 1, i);
                Dl vb = dmin0((wreal)0.5 * (rv0 + rv1));
                ty = dy_inv * (vb * (Q(k, 1, i) - Q(k, 0, i)) + Q(k, 0, i) * (rv1 - rv0));
                ay = true;
            } else {
                Dl rv0 = RV(k, ny - 1, i), rv1 = RV(k, ny, i);
                Dl vb = dmax0((wreal)0.5 * (rv0 + rv1));
                ty = dy_inv * (vb * (Q(k, ny - 1, i) - Q(k, ny - 2, i))
                               + Q(k, ny - 1, i) * (rv1 - rv0));
                ay = true;
            }
        } else {
            Dl fy0 = ty_cell(q, qd, RV(k, j, i), k, j, i, ny, nyq, nxs, open_y);
            Dl fy1 = ty_cell(q, qd, RV(k, j + 1, i), k, j + 1, i, ny, nyq, nxs, open_y);
            ty = (mscale * dy_inv) * (fy1 - fy0);
            ay = true;
        }
        Dl fz0 = tz_half(q, qd, RW(k, j, i), k, j, i, nz, nyq, nxs, fnm, fnp, vorder);
        Dl fz1 = tz_half(q, qd, RW(k + 1, j, i), k + 1, j, i, nz, nyq, nxs, fnm, fnp, vorder);
        Dl t = ld(tend, tendd, at);
#if ADV_X_FIRST
        if (ax) t = t - tx;
        if (ay) t = t - ty;
#else
        if (ay) t = t - ty;
        if (ax) t = t - tx;
#endif
        res = t - rdz[k] * (fz1 - fz0);
    } else {
        Dl fx[2], fy[2], fz[2];
        for (int s = 0; s < 2; ++s) {
            fx[s] = tx_cell(q, qd, RU(k, j, i + s), k, j, i + s, nx, nyq, nxs, 0);
            fy[s] = ty_cell(q, qd, RV(k, j + s, i), k, j + s, i, ny, nyq, nxs, 0);
            fz[s] = tz_half(q, qd, RW(k + s, j, i), k + s, j, i, nz, nyq, nxs, fnm, fnp, vorder);
        }
        Dl r = ld(tend, tendd, at);
#if ADV_X_FIRST
        r = r - (mscale * dx_inv) * (fx[1] - fx[0]);
        r = r - (mscale * dy_inv) * (fy[1] - fy[0]);
#else
        r = r - (mscale * dy_inv) * (fy[1] - fy[0]);
        r = r - (mscale * dx_inv) * (fx[1] - fx[0]);
#endif
        res = r - rdz[k] * (fz[1] - fz[0]);
    }
#undef Q
    if (WRFPLUS_TL) tendd[at] = res.d;
    if (nl) tend[at] = res.v;
}

// ----------------------------------------------------------------- u TL

extern "C" __global__ void adv_u_tl(TL_ARGS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;     // u-point 0..nx
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx + 1 || j >= ny || k >= nz) return;
    const int nxs = nx + 1, nyq = ny;
    wreal mscale = msf[(size_t)j * (nx + 1) + i];
    size_t at = R3(k, j, i, ny, nx + 1);
    Dl res;
#define U(kk, jj, ii) ld(q, qd, R3((kk), (jj), (ii), ny, nx + 1))
    if (open_x || open_y) {
        bool xedge = open_x && (i == 0 || i == nx);
        int cl = open_x ? max(i - 1, 0) : awrap(i - 1, nx);
        int cr = open_x ? min(i, nx - 1) : awrap(i, nx);
        int iu = open_x ? i : cr;
        Dl tx = dl(0, 0), ty = dl(0, 0);
        bool ax = false, ay = false;
        if (!xedge) {
            Dl fx2[2];
            for (int s = 0; s < 2; ++s) {
                int m = (s == 0) ? cl : cr;
                Dl velx = (wreal)0.5 * (RU(k, j, m) + RU(k, j, m + 1));
                if (open_x) {
                    int d = min(m, nx - 1 - m);
                    if (d == 0) {
                        Dl qa = U(k, j, m), qb = U(k, j, m + 1);
                        if (spec) {
                            if (m == 0) { if (qb.v < (wreal)0) qa = qb; }
                            else        { if (qa.v > (wreal)0) qb = qa; }
                        }
                        fx2[s] = tf2(qa, qb, velx);
                    } else if (d == 1) {
                        fx2[s] = tf3h(U(k, j, m - 1), U(k, j, m), U(k, j, m + 1), U(k, j, m + 2), velx);
                    } else {
                        fx2[s] = tfh(U(k, j, m - 2), U(k, j, m - 1), U(k, j, m), U(k, j, m + 1),
                                     U(k, j, m + 2), U(k, j, m + 3), velx);
                    }
                } else {
                    fx2[s] = tfh(U(k, j, awrap(m - 2, nx)), U(k, j, awrap(m - 1, nx)),
                                 U(k, j, awrap(m, nx)), U(k, j, awrap(m + 1, nx)),
                                 U(k, j, awrap(m + 2, nx)), U(k, j, awrap(m + 3, nx)), velx);
                }
            }
            tx = (mscale * dx_inv) * (fx2[1] - fx2[0]);
            ax = true;
        }
        if (!xedge) {
            if (open_y && (j == 0 || j == ny - 1)) {
                if (spec) {
                } else if (j == 0) {
                    Dl vw = (wreal)0.5 * (RV(k, 0, cl) + RV(k, 0, cr));
                    Dl vb = dmin0(vw);
                    Dl dsum = (RV(k, 1, cl) - RV(k, 0, cl)) + (RV(k, 1, cr) - RV(k, 0, cr));
                    ty = (mscale * dy_inv) * (vb * (U(k, 1, iu) - U(k, 0, iu))
                                              + ((wreal)0.5 * U(k, 0, iu)) * dsum);
                    ay = true;
                } else {
                    Dl vw = (wreal)0.5 * (RV(k, ny, cl) + RV(k, ny, cr));
                    Dl vb = dmax0(vw);
                    Dl dsum = (RV(k, ny, cl) - RV(k, ny - 1, cl)) + (RV(k, ny, cr) - RV(k, ny - 1, cr));
                    ty = (mscale * dy_inv) * (vb * (U(k, ny - 1, iu) - U(k, ny - 2, iu))
                                              + ((wreal)0.5 * U(k, ny - 1, iu)) * dsum);
                    ay = true;
                }
            } else {
                Dl fy2[2];
                for (int s = 0; s < 2; ++s) {
                    int g = j + s;
                    Dl vely = (wreal)0.5 * (RV(k, g, cl) + RV(k, g, cr));
                    fy2[s] = ty_cell(q, qd, vely, k, g, iu, ny, nyq, nxs, open_y);
                }
                ty = (mscale * dy_inv) * (fy2[1] - fy2[0]);
                ay = true;
            }
        }
        Dl t = ld(tend, tendd, at);
#if ADV_X_FIRST
        if (ax) t = t - tx;
        if (ay) t = t - ty;
#else
        if (ay) t = t - ty;
        if (ax) t = t - tx;
#endif
        if (!(xedge && open_y)) {
            Dl fz2[2];
            for (int s = 0; s < 2; ++s) {
                int kf = k + s;
                Dl velz;
                if (open_x && i == 0) velz = RW(kf, j, 0);
                else if (open_x && i == nx) velz = RW(kf, j, nx - 1);
                else velz = (wreal)0.5 * (RW(kf, j, cl) + RW(kf, j, cr));
                fz2[s] = tz_half(q, qd, velz, kf, j, iu, nz, nyq, nxs, fnm, fnp, vorder);
            }
            t = t - rdz[k] * (fz2[1] - fz2[0]);
        }
        res = t;
    } else {
        int ic = awrap(i, nx), im1c = awrap(i - 1, nx);
        Dl fx[2], fy[2], fz[2];
        for (int s = 0; s < 2; ++s) {
            int m = (s == 0) ? im1c : ic;
            Dl velx = (wreal)0.5 * (RU(k, j, m) + RU(k, j, m + 1));
            fx[s] = tfh(U(k, j, awrap(m - 2, nx)), U(k, j, awrap(m - 1, nx)), U(k, j, awrap(m, nx)),
                        U(k, j, awrap(m + 1, nx)), U(k, j, awrap(m + 2, nx)),
                        U(k, j, awrap(m + 3, nx)), velx);
            int g = j + s;
            Dl vely = (wreal)0.5 * (RV(k, g, im1c) + RV(k, g, ic));
            fy[s] = ty_cell(q, qd, vely, k, g, ic, ny, nyq, nxs, 0);
            int kf = k + s;
            Dl velz = (wreal)0.5 * (RW(kf, j, im1c) + RW(kf, j, ic));
            fz[s] = tz_half(q, qd, velz, kf, j, ic, nz, nyq, nxs, fnm, fnp, vorder);
        }
        Dl r = ld(tend, tendd, at);
#if ADV_X_FIRST
        r = r - (mscale * dx_inv) * (fx[1] - fx[0]);
        r = r - (mscale * dy_inv) * (fy[1] - fy[0]);
#else
        r = r - (mscale * dy_inv) * (fy[1] - fy[0]);
        r = r - (mscale * dx_inv) * (fx[1] - fx[0]);
#endif
        res = r - rdz[k] * (fz[1] - fz[0]);
    }
#undef U
    if (WRFPLUS_TL) tendd[at] = res.d;
    if (nl) tend[at] = res.v;
}

// ----------------------------------------------------------------- v TL

extern "C" __global__ void adv_v_tl(TL_ARGS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;     // v-point 0..ny
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny + 1 || k >= nz) return;
    const int nyq = ny + 1, nxs = nx;
    wreal mscale = msf[(size_t)j * nx + i];
    size_t at = R3(k, j, i, ny + 1, nx);
    Dl res;
#define V(kk, jj, ii) ld(q, qd, R3((kk), (jj), (ii), ny + 1, nx))
    if (open_x || open_y) {
        bool yedge = open_y && (j == 0 || j == ny);
        int rs = open_y ? max(j - 1, 0) : awrap(j - 1, ny);
        int rn = open_y ? min(j, ny - 1) : awrap(j, ny);
        int jv = open_y ? j : rn;
        Dl tx = dl(0, 0), ty = dl(0, 0);
        bool ax = false, ay = false;
        if (!yedge) {
            Dl fy2[2];
            for (int s = 0; s < 2; ++s) {
                int m = (s == 0) ? rs : rn;
                Dl vely = (wreal)0.5 * (RV(k, m, i) + RV(k, m + 1, i));
                if (open_y) {
                    int d = min(m, ny - 1 - m);
                    if (d == 0) {
                        Dl qa = V(k, m, i), qb = V(k, m + 1, i);
                        if (spec) {
                            if (m == 0) { if (qb.v < (wreal)0) qa = qb; }
                            else        { if (qa.v > (wreal)0) qb = qa; }
                        }
                        fy2[s] = tf2(qa, qb, vely);
                    } else if (d == 1) {
                        fy2[s] = tf3h(V(k, m - 1, i), V(k, m, i), V(k, m + 1, i), V(k, m + 2, i), vely);
                    } else {
                        fy2[s] = tfh(V(k, m - 2, i), V(k, m - 1, i), V(k, m, i), V(k, m + 1, i),
                                     V(k, m + 2, i), V(k, m + 3, i), vely);
                    }
                } else {
                    fy2[s] = tfh(V(k, awrap(m - 2, ny), i), V(k, awrap(m - 1, ny), i),
                                 V(k, awrap(m, ny), i), V(k, awrap(m + 1, ny), i),
                                 V(k, awrap(m + 2, ny), i), V(k, awrap(m + 3, ny), i), vely);
                }
            }
            ty = (mscale * dy_inv) * (fy2[1] - fy2[0]);
            ay = true;
        }
        if (!yedge) {
            if (open_x && (i == 0 || i == nx - 1)) {
                if (spec) {
                } else if (i == 0) {
                    Dl uw0 = (wreal)0.5 * (RU(k, rs, 0) + RU(k, rn, 0));
                    Dl uw1 = (wreal)0.5 * (RU(k, rs, 1) + RU(k, rn, 1));
                    Dl ub = dmin0(uw0);
                    tx = (mscale * dx_inv) * (ub * (V(k, jv, 1) - V(k, jv, 0))
                                              + V(k, jv, 0) * (uw1 - uw0));
                    ax = true;
                } else {
                    Dl uw0 = (wreal)0.5 * (RU(k, rs, nx - 1) + RU(k, rn, nx - 1));
                    Dl uw1 = (wreal)0.5 * (RU(k, rs, nx) + RU(k, rn, nx));
                    Dl ub = dmax0(uw1);
                    tx = (mscale * dx_inv) * (ub * (V(k, jv, nx - 1) - V(k, jv, nx - 2))
                                              + V(k, jv, nx - 1) * (uw1 - uw0));
                    ax = true;
                }
            } else {
                Dl fx2[2];
                for (int s = 0; s < 2; ++s) {
                    int f = i + s;
                    Dl velx = (wreal)0.5 * (RU(k, rs, f) + RU(k, rn, f));
                    fx2[s] = tx_cell(q, qd, velx, k, jv, f, nx, nyq, nxs, open_x);
                }
                tx = (mscale * dx_inv) * (fx2[1] - fx2[0]);
                ax = true;
            }
        }
        Dl t = ld(tend, tendd, at);
        if (ay) t = t - ty;
        if (ax) t = t - tx;
        if (!yedge) {
            Dl fz2[2];
            for (int s = 0; s < 2; ++s) {
                int kf = k + s;
                Dl velz = (wreal)0.5 * (RW(kf, rs, i) + RW(kf, rn, i));
                fz2[s] = tz_half(q, qd, velz, kf, jv, i, nz, nyq, nxs, fnm, fnp, vorder);
            }
            t = t - rdz[k] * (fz2[1] - fz2[0]);
        }
        res = t;
    } else {
        int jc = awrap(j, ny), jm1c = awrap(j - 1, ny);
        Dl fx[2], fy[2], fz[2];
        for (int s = 0; s < 2; ++s) {
            int f = i + s;
            Dl velx = (wreal)0.5 * (RU(k, jm1c, f) + RU(k, jc, f));
            fx[s] = tx_cell(q, qd, velx, k, jc, f, nx, nyq, nxs, 0);
            int m = (s == 0) ? jm1c : jc;
            Dl vely = (wreal)0.5 * (RV(k, m, i) + RV(k, m + 1, i));
            fy[s] = tfh(V(k, awrap(m - 2, ny), i), V(k, awrap(m - 1, ny), i), V(k, awrap(m, ny), i),
                        V(k, awrap(m + 1, ny), i), V(k, awrap(m + 2, ny), i),
                        V(k, awrap(m + 3, ny), i), vely);
            int kf = k + s;
            Dl velz = (wreal)0.5 * (RW(kf, jm1c, i) + RW(kf, jc, i));
            fz[s] = tz_half(q, qd, velz, kf, jc, i, nz, nyq, nxs, fnm, fnp, vorder);
        }
        Dl r = ld(tend, tendd, at);
        r = r - (mscale * dy_inv) * (fy[1] - fy[0]);
        r = r - (mscale * dx_inv) * (fx[1] - fx[0]);
        res = r - rdz[k] * (fz[1] - fz[0]);
    }
#undef V
    if (WRFPLUS_TL) tendd[at] = res.d;
    if (nl) tend[at] = res.v;
}

// ----------------------------------------------------------------- w TL

// advect_w's lid face winds: (2 - fzm) * F(ktf) - fzp * F(ktf-1)
static __device__ __forceinline__ Dl lid_vel(Dl top, Dl below, const wreal* fnm,
                                             const wreal* fnp, int nz)
{
    return ((wreal)2 - fnm[nz - 1]) * top - fnp[nz - 1] * below;
}

extern "C" __global__ void adv_w_tl(TL_ARGS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;     // w-level 0..nz
    if (i >= nx || j >= ny || k > nz) return;
    if (k == 0 || (k == nz && nz < 2)) return;
    const int nyq = ny, nxs = nx;
    wreal mscale = msf[(size_t)j * nx + i];
    size_t at = R3(k, j, i, ny, nx);
    Dl res;
#define W(kk, jj, ii) ld(q, qd, R3((kk), (jj), (ii), ny, nx))
    if (k == nz) {
        // The lid's open non-cb terms are formed but never applied in the
        // strict primal (no applyx/applyy), so they are absent here.
        Dl tx = dl(0, 0), ty = dl(0, 0);
        bool ax = false, ay = false;
        if (!(open_x && (i == 0 || i == nx - 1))) {
            Dl r0 = lid_vel(RU(nz - 1, j, i), RU(nz - 2, j, i), fnm, fnp, nz);
            Dl r1 = lid_vel(RU(nz - 1, j, i + 1), RU(nz - 2, j, i + 1), fnm, fnp, nz);
            Dl f0 = tx_cell(q, qd, r0, k, j, i, nx, nyq, nxs, open_x);
            Dl f1 = tx_cell(q, qd, r1, k, j, i + 1, nx, nyq, nxs, open_x);
            tx = (mscale * dx_inv) * (f1 - f0);
            ax = true;
        }
        if (!(open_y && (j == 0 || j == ny - 1))) {
            Dl r0 = lid_vel(RV(nz - 1, j, i), RV(nz - 2, j, i), fnm, fnp, nz);
            Dl r1 = lid_vel(RV(nz - 1, j + 1, i), RV(nz - 2, j + 1, i), fnm, fnp, nz);
            Dl f0 = ty_cell(q, qd, r0, k, j, i, ny, nyq, nxs, open_y);
            Dl f1 = ty_cell(q, qd, r1, k, j + 1, i, ny, nyq, nxs, open_y);
            ty = (mscale * dy_inv) * (f1 - f0);
            ay = true;
        }
        Dl h = ld(tend, tendd, at);
#if ADV_X_FIRST
        if (ax) h = h - tx;
        if (ay) h = h - ty;
#else
        if (ay) h = h - ty;
        if (ax) h = h - tx;
#endif
        Dl top_flux = ((wreal)0.25 * (RW(k, j, i) + RW(k - 1, j, i))) * (W(k, j, i) + W(k - 1, j, i));
        res = h + ((wreal)2 * rdz[k - 1]) * top_flux;
    } else if (open_x || open_y) {
        Dl tx = dl(0, 0), ty = dl(0, 0);
        bool ax = false, ay = false;
        if (open_x && (i == 0 || i == nx - 1)) {
            if (spec) {
            } else if (i == 0) {
                Dl ruw0 = fnm[k] * RU(k, j, 0) + fnp[k] * RU(k - 1, j, 0);
                Dl ruw1 = fnm[k] * RU(k, j, 1) + fnp[k] * RU(k - 1, j, 1);
                Dl ub = dmin0((wreal)0.5 * (ruw0 + ruw1));
                tx = dx_inv * (ub * (W(k, j, 1) - W(k, j, 0)) + W(k, j, 0) * (ruw1 - ruw0));
                ax = true;
            } else {
                Dl ruw0 = fnm[k] * RU(k, j, nx - 1) + fnp[k] * RU(k - 1, j, nx - 1);
                Dl ruw1 = fnm[k] * RU(k, j, nx) + fnp[k] * RU(k - 1, j, nx);
                Dl ub = dmax0((wreal)0.5 * (ruw0 + ruw1));
                tx = dx_inv * (ub * (W(k, j, nx - 1) - W(k, j, nx - 2))
                               + W(k, j, nx - 1) * (ruw1 - ruw0));
                ax = true;
            }
        } else {
            Dl fx2[2];
            for (int s = 0; s < 2; ++s) {
                int f = i + s;
                Dl velx = fnm[k] * RU(k, j, f) + fnp[k] * RU(k - 1, j, f);
                fx2[s] = tx_cell(q, qd, velx, k, j, f, nx, nyq, nxs, open_x);
            }
            tx = (mscale * dx_inv) * (fx2[1] - fx2[0]);
            ax = true;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
            } else if (j == 0) {
                Dl rvw0 = fnm[k] * RV(k, 0, i) + fnp[k] * RV(k - 1, 0, i);
                Dl rvw1 = fnm[k] * RV(k, 1, i) + fnp[k] * RV(k - 1, 1, i);
                Dl vb = dmin0((wreal)0.5 * (rvw0 + rvw1));
                ty = dy_inv * (vb * (W(k, 1, i) - W(k, 0, i)) + W(k, 0, i) * (rvw1 - rvw0));
                ay = true;
            } else {
                Dl rvw0 = fnm[k] * RV(k, ny - 1, i) + fnp[k] * RV(k - 1, ny - 1, i);
                Dl rvw1 = fnm[k] * RV(k, ny, i) + fnp[k] * RV(k - 1, ny, i);
                Dl vb = dmax0((wreal)0.5 * (rvw0 + rvw1));
                ty = dy_inv * (vb * (W(k, ny - 1, i) - W(k, ny - 2, i))
                               + W(k, ny - 1, i) * (rvw1 - rvw0));
                ay = true;
            }
        } else {
            Dl fy2[2];
            for (int s = 0; s < 2; ++s) {
                int g = j + s;
                Dl vely = fnm[k] * RV(k, g, i) + fnp[k] * RV(k - 1, g, i);
                fy2[s] = ty_cell(q, qd, vely, k, g, i, ny, nyq, nxs, open_y);
            }
            ty = (mscale * dy_inv) * (fy2[1] - fy2[0]);
            ay = true;
        }
        Dl fz2[2];
        for (int s = 0; s < 2; ++s) {
            int m = k - 1 + s;
            Dl velz = (wreal)0.5 * (RW(m, j, i) + RW(m + 1, j, i));
            fz2[s] = tz_w(q, qd, velz, m, j, i, nz, nyq, nxs, vorder);
        }
        Dl t = ld(tend, tendd, at);
#if ADV_X_FIRST
        if (ax) t = t - tx;
        if (ay) t = t - ty;
#else
        if (ay) t = t - ty;
        if (ax) t = t - tx;
#endif
        res = t - rdz[k] * (fz2[1] - fz2[0]);
    } else {
        Dl fx[2], fy[2], fz[2];
        for (int s = 0; s < 2; ++s) {
            int f = i + s;
            Dl velx = fnm[k] * RU(k, j, f) + fnp[k] * RU(k - 1, j, f);
            fx[s] = tx_cell(q, qd, velx, k, j, f, nx, nyq, nxs, 0);
            int g = j + s;
            Dl vely = fnm[k] * RV(k, g, i) + fnp[k] * RV(k - 1, g, i);
            fy[s] = ty_cell(q, qd, vely, k, g, i, ny, nyq, nxs, 0);
            int m = k - 1 + s;
            Dl velz = (wreal)0.5 * (RW(m, j, i) + RW(m + 1, j, i));
            fz[s] = tz_w(q, qd, velz, m, j, i, nz, nyq, nxs, vorder);
        }
        Dl r = ld(tend, tendd, at);
#if ADV_X_FIRST
        r = r - (mscale * dx_inv) * (fx[1] - fx[0]);
        r = r - (mscale * dy_inv) * (fy[1] - fy[0]);
#else
        r = r - (mscale * dy_inv) * (fy[1] - fy[0]);
        r = r - (mscale * dx_inv) * (fx[1] - fx[0]);
#endif
        res = r - rdz[k] * (fz[1] - fz[0]);
    }
#undef W
    if (WRFPLUS_TL) tendd[at] = res.d;
    if (nl) tend[at] = res.v;
}

#undef RU
#undef RV
#undef RW

// ===================================================================== AD
//
// Face flux types.  Slots s = 0..5 are (qm3, qm2, qm1, q0, qp1, qp2) about
// the face's q0 point; F = V * sum(c q) + sg * |V| * sum(e q).

#define FT_ZERO 0
#define FT_F2 1     // flux2
#define FT_Q0 2     // specified upstream substitution: V * q0
#define FT_QM1 3    // specified upstream substitution: V * qm1
#define FT_F3H 4    // horizontal flux3 (flux3h)
#define FT_FH 5     // horizontal flux5 macro (ADV_H)
#define FT_Z2 6     // V * (fnm q0 + fnp qm1)
#define FT_F3V 7    // vertical flux3
#define FT_F4 8
#define FT_F5V 9    // vertical flux5
#define FT_F6 10

struct FW { wreal c0, c1, c2, c3, c4, c5, e0, e1, e2, e3, e4, e5, sg; };

static __device__ __forceinline__ FW fw_zero()
{
    FW w;
    w.c0 = w.c1 = w.c2 = w.c3 = w.c4 = w.c5 = (wreal)0;
    w.e0 = w.e1 = w.e2 = w.e3 = w.e4 = w.e5 = (wreal)0;
    w.sg = (wreal)0;
    return w;
}
static __device__ __forceinline__ void fw_c4(FW& w)
{
    w.c1 = (wreal)-1 / (wreal)12; w.c2 = (wreal)7 / (wreal)12;
    w.c3 = (wreal)7 / (wreal)12;  w.c4 = (wreal)-1 / (wreal)12;
}
static __device__ __forceinline__ void fw_e3(FW& w, wreal sg)
{
    w.e1 = (wreal)-1 / (wreal)12; w.e2 = (wreal)3 / (wreal)12;
    w.e3 = (wreal)-3 / (wreal)12; w.e4 = (wreal)1 / (wreal)12;
    w.sg = sg;
}
static __device__ __forceinline__ void fw_c6(FW& w)
{
    w.c0 = (wreal)1 / (wreal)60;  w.c1 = (wreal)-8 / (wreal)60; w.c2 = (wreal)37 / (wreal)60;
    w.c3 = (wreal)37 / (wreal)60; w.c4 = (wreal)-8 / (wreal)60; w.c5 = (wreal)1 / (wreal)60;
}
static __device__ __forceinline__ void fw_e5(FW& w, wreal sg)
{
    w.e0 = (wreal)-1 / (wreal)60;  w.e1 = (wreal)5 / (wreal)60; w.e2 = (wreal)-10 / (wreal)60;
    w.e3 = (wreal)10 / (wreal)60;  w.e4 = (wreal)-5 / (wreal)60; w.e5 = (wreal)1 / (wreal)60;
    w.sg = sg;
}
static __device__ FW fw_make(int t, wreal zm, wreal zp)
{
    FW w = fw_zero();
    switch (t) {
    case FT_F2: w.c2 = (wreal)0.5; w.c3 = (wreal)0.5; break;
    case FT_Q0: w.c3 = (wreal)1; break;
    case FT_QM1: w.c2 = (wreal)1; break;
    case FT_F3H:
        fw_c4(w);
#if !(ADV_H == 4 || ADV_H == 6)
        fw_e3(w, (wreal)1);
#endif
        break;
    case FT_FH:
#if ADV_H == 3
        fw_c4(w); fw_e3(w, (wreal)1);
#elif ADV_H == 4
        fw_c4(w);
#elif ADV_H == 6
        fw_c6(w);
#else
        fw_c6(w); fw_e5(w, (wreal)-1);
#endif
        break;
    case FT_Z2: w.c2 = zp; w.c3 = zm; break;
    case FT_F3V: fw_c4(w); fw_e3(w, (wreal)-1); break;
    case FT_F4: fw_c4(w); break;
    case FT_F5V: fw_c6(w); fw_e5(w, (wreal)1); break;
    case FT_F6: fw_c6(w); break;
    default: break;
    }
    return w;
}
static __device__ __forceinline__ wreal fw_c(FW w, int s)
{
    return s == 0 ? w.c0 : s == 1 ? w.c1 : s == 2 ? w.c2 : s == 3 ? w.c3 : s == 4 ? w.c4 : w.c5;
}
static __device__ __forceinline__ wreal fw_e(FW w, int s)
{
    return s == 0 ? w.e0 : s == 1 ? w.e1 : s == 2 ? w.e2 : s == 3 ? w.e3 : s == 4 ? w.e4 : w.e5;
}
// dF/dV given the slot values
static __device__ __forceinline__ wreal fw_v(FW w, wreal V, wreal q0, wreal q1, wreal q2,
                                             wreal q3, wreal q4, wreal q5)
{
    wreal sv = w.sg * wsign(V);
    return (w.c0 + sv * w.e0) * q0 + (w.c1 + sv * w.e1) * q1 + (w.c2 + sv * w.e2) * q2
         + (w.c3 + sv * w.e3) * q3 + (w.c4 + sv * w.e4) * q4 + (w.c5 + sv * w.e5) * q5;
}

// The stencil coefficients by slot (constants once S is a template
// argument), and dF/dq(slot S) of face type t without the FW table.
static __device__ __forceinline__ wreal k_c4(int s)
{
    return (s == 1 || s == 4) ? (wreal)-1 / (wreal)12
         : (s == 2 || s == 3) ? (wreal)7 / (wreal)12 : (wreal)0;
}
static __device__ __forceinline__ wreal k_e3(int s)
{
    return s == 1 ? (wreal)-1 / (wreal)12 : s == 2 ? (wreal)3 / (wreal)12
         : s == 3 ? (wreal)-3 / (wreal)12 : s == 4 ? (wreal)1 / (wreal)12 : (wreal)0;
}
static __device__ __forceinline__ wreal k_c6(int s)
{
    return (s == 0 || s == 5) ? (wreal)1 / (wreal)60 : (s == 1 || s == 4) ? (wreal)-8 / (wreal)60
         : (wreal)37 / (wreal)60;
}
static __device__ __forceinline__ wreal k_e5(int s)
{
    return s == 0 ? (wreal)-1 / (wreal)60 : s == 1 ? (wreal)5 / (wreal)60
         : s == 2 ? (wreal)-10 / (wreal)60 : s == 3 ? (wreal)10 / (wreal)60
         : s == 4 ? (wreal)-5 / (wreal)60 : (wreal)1 / (wreal)60;
}
static __device__ __forceinline__ wreal fw_qs(int S, int t, wreal V, wreal zm, wreal zp)
{
    wreal a = fabs(V);
    switch (t) {
    case FT_F2: return (S == 2 || S == 3) ? (wreal)0.5 * V : (wreal)0;
    case FT_Q0: return S == 3 ? V : (wreal)0;
    case FT_QM1: return S == 2 ? V : (wreal)0;
#if ADV_H == 4 || ADV_H == 6
    case FT_F3H: return V * k_c4(S);
#else
    case FT_F3H: return V * k_c4(S) + a * k_e3(S);
#endif
#if ADV_H == 3
    case FT_FH: return V * k_c4(S) + a * k_e3(S);
#elif ADV_H == 4
    case FT_FH: return V * k_c4(S);
#elif ADV_H == 6
    case FT_FH: return V * k_c6(S);
#else
    case FT_FH: return V * k_c6(S) - a * k_e5(S);
#endif
    case FT_Z2: return S == 2 ? V * zp : S == 3 ? V * zm : (wreal)0;
    case FT_F3V: return V * k_c4(S) - a * k_e3(S);
    case FT_F4: return V * k_c4(S);
    case FT_F5V: return V * k_c6(S) + a * k_e5(S);
    case FT_F6: return V * k_c6(S);
    default: return 0;
    }
}

// ------------------------------------------------- cell-type face lines
//
// One line of cells c = 0..n-1 along x or y with faces f = 0..n between
// them (face f below cell f): scalar and w along x and y, u along y, v
// along x.  q, the tendency adjoint and msf are read at base + c*stride;
// the face's advecting flux is V(f) = a1*vel[vb1 + f*vs] + a2*vel[vb2 + f*vs].
// flav: 0 no non-cb terms; 1 scalar/w (ub from the mean of the two face
// fluxes, plain dinv); 2 u/v (ub from the boundary face's flux, msf*dinv).

struct CLine {
    size_t qb, qs, tb, ts, mb, ms, vb1, vb2, vs;
    wreal a1, a2, dinv;
    int n, open, flav, on;
};

static __device__ __forceinline__ wreal cl_V(const wreal* vel, CLine L, int f)
{
    return L.a1 * vel[L.vb1 + (size_t)f * L.vs] + L.a2 * vel[L.vb2 + (size_t)f * L.vs];
}
static __device__ __forceinline__ int cl_type(CLine L, int f)
{
    if (!L.open) return FT_FH;
    if (f <= 0 || f >= L.n) return FT_ZERO;
    int d = min(f, L.n - f);
    return d == 1 ? FT_F2 : d == 2 ? FT_F3H : FT_FH;
}
static __device__ __forceinline__ bool cl_flux(CLine L, int c)
{
    return L.on && !(L.open && (c == 0 || c == L.n - 1));
}
static __device__ wreal cl_aF(const wreal* at, const wreal* msf, CLine L, int f)
{
    wreal r = 0;
    if (f < L.n && cl_flux(L, f))
        r += (msf[L.mb + (size_t)f * L.ms] * L.dinv) * at[L.tb + (size_t)f * L.ts];
    if (f >= 1 && cl_flux(L, f - 1))
        r -= (msf[L.mb + (size_t)(f - 1) * L.ms] * L.dinv) * at[L.tb + (size_t)(f - 1) * L.ts];
    return r;
}
// q at cell c (wrapped when periodic); 0 outside an open line
static __device__ __forceinline__ wreal cl_q(const wreal* q, CLine L, int c)
{
    if (!L.open) c = awrap(c, L.n);
    else if (c < 0 || c >= L.n) return 0;
    return q[L.qb + (size_t)c * L.qs];
}
// Non-cb boundary cell c0 (0 or n-1):
//   t -= cf * (ub * (q[a] - q[b]) + q[e] * (V1 - V0)),
//   c0 = 0:   a = 1, b = e = 0, V0 = V(0), V1 = V(1), ub = min(x, 0);
//   c0 = n-1: a = e = n-1, b = n-2, V0 = V(n-1), V1 = V(n), ub = max(x, 0);
//   x = (V0 + V1)/2 (flav 1) or V0 / V1 (flav 2, the boundary face's flux).
static __device__ wreal cl_ncb_A(const wreal* at, const wreal* msf, CLine L, int c0)
{
    wreal cf = L.flav == 2 ? msf[L.mb + (size_t)c0 * L.ms] * L.dinv : L.dinv;
    return -cf * at[L.tb + (size_t)c0 * L.ts];
}
static __device__ __forceinline__ wreal cl_ncb_x(CLine L, int c0, wreal V0, wreal V1)
{
    if (L.flav == 1) return (wreal)0.5 * (V0 + V1);
    return c0 == 0 ? V0 : V1;
}
static __device__ __forceinline__ wreal cl_ncb_ub(int c0, wreal x)
{
    return c0 == 0 ? wmin0(x) : wmax0(x);
}
// adjoint into q at cell p from the line's non-cb terms
static __device__ wreal cl_ncb_q(const wreal* at, const wreal* msf, const wreal* vel, CLine L,
                                 int p, int spec)
{
    if (!(L.flav && L.on && L.open && !spec)) return 0;
    wreal r = 0;
    for (int side = 0; side < 2; ++side) {
        int c0 = side == 0 ? 0 : L.n - 1;
        int fa = side == 0 ? 0 : L.n - 1;
        int ia = side == 0 ? 1 : L.n - 1, ib = side == 0 ? 0 : L.n - 2;
        int ie = side == 0 ? 0 : L.n - 1;
        if (p != ia && p != ib && p != ie) continue;
        wreal V0 = cl_V(vel, L, fa), V1 = cl_V(vel, L, fa + 1);
        wreal ub = cl_ncb_ub(c0, cl_ncb_x(L, c0, V0, V1));
        wreal A = cl_ncb_A(at, msf, L, c0);
        if (p == ia) r += A * ub;
        if (p == ib) r -= A * ub;
        if (p == ie) r += A * (V1 - V0);
    }
    return r;
}
// adjoint into V(f) from the line's non-cb terms
static __device__ wreal cl_ncb_V(const wreal* q, const wreal* at, const wreal* msf,
                                 const wreal* vel, CLine L, int f, int spec)
{
    if (!(L.flav && L.on && L.open && !spec)) return 0;
    wreal r = 0;
    for (int side = 0; side < 2; ++side) {
        int c0 = side == 0 ? 0 : L.n - 1;
        int fa = side == 0 ? 0 : L.n - 1;
        if (f != fa && f != fa + 1) continue;
        int ia = side == 0 ? 1 : L.n - 1, ib = side == 0 ? 0 : L.n - 2;
        int ie = side == 0 ? 0 : L.n - 1;
        wreal V0 = cl_V(vel, L, fa), V1 = cl_V(vel, L, fa + 1);
        wreal x = cl_ncb_x(L, c0, V0, V1);
        bool live = c0 == 0 ? x < (wreal)0 : x > (wreal)0;
        wreal A = cl_ncb_A(at, msf, L, c0);
        wreal ax = live ? A * (cl_q(q, L, ia) - cl_q(q, L, ib)) : (wreal)0;
        wreal qe = cl_q(q, L, ie);
        wreal a0, a1;
        if (L.flav == 1) {
            a0 = (wreal)0.5 * ax;
            a1 = (wreal)0.5 * ax;
        } else {
            a0 = c0 == 0 ? ax : (wreal)0;
            a1 = c0 == 0 ? (wreal)0 : ax;
        }
        if (f == fa) r += a0 - A * qe;
        if (f == fa + 1) r += a1 + A * qe;
    }
    return r;
}
static __device__ __forceinline__ wreal cl_face(int S, const wreal* at, const wreal* msf,
                                                const wreal* vel, CLine L, int f)
{
    int t = cl_type(L, f);
    if (t == FT_ZERO) return 0;
    return cl_aF(at, msf, L, f) * fw_qs(S, t, cl_V(vel, L, f), 0, 0);
}
// the face whose slot S is cell p (and, periodic, its copy at face n)
static __device__ __forceinline__ wreal cl_slot(int S, const wreal* at, const wreal* msf,
                                                const wreal* vel, CLine L, int p)
{
    int f = p + 3 - S;
    if (L.open) return (f < 0 || f > L.n) ? (wreal)0 : cl_face(S, at, msf, vel, L, f);
    f = awrap(f, L.n);
    wreal r = cl_face(S, at, msf, vel, L, f);
    if (f == 0) r += cl_face(S, at, msf, vel, L, L.n);
    return r;
}
// adjoint into q at cell p (0..n-1) from the line's face fluxes and non-cb terms
static __device__ wreal cl_aq(const wreal* at, const wreal* msf, const wreal* vel, CLine L,
                              int p, int spec)
{
    if (!L.on) return 0;
    wreal acc = cl_slot(0, at, msf, vel, L, p) + cl_slot(1, at, msf, vel, L, p)
              + cl_slot(2, at, msf, vel, L, p) + cl_slot(3, at, msf, vel, L, p)
              + cl_slot(4, at, msf, vel, L, p) + cl_slot(5, at, msf, vel, L, p);
    return acc + cl_ncb_q(at, msf, vel, L, p, spec);
}
// adjoint into the line's face flux V(f), f = 0..n
static __device__ wreal cl_aV(const wreal* q, const wreal* at, const wreal* msf,
                              const wreal* vel, CLine L, int f, int spec)
{
    if (!L.on) return 0;
    wreal r = 0;
    int t = cl_type(L, f);
    if (t != FT_ZERO) {
        wreal aF = cl_aF(at, msf, L, f);
        FW w = fw_make(t, 0, 0);
        r = aF * fw_v(w, cl_V(vel, L, f), cl_q(q, L, f - 3), cl_q(q, L, f - 2), cl_q(q, L, f - 1),
                      cl_q(q, L, f), cl_q(q, L, f + 1), cl_q(q, L, f + 2));
    }
    return r + cl_ncb_V(q, at, msf, vel, L, f, spec);
}

// ----------------------------------------------- mass-centre face lines
//
// u along x and v along y: points c = 0..n (the staggered field), faces
// m = 0..n-1 at the mass centres with V(m) = (vel[m] + vel[m+1]) / 2 and
// q0 point m+1.  Point c takes faces cl(c) (below) and cr(c) (above); an
// open line's end points take no flux along the line.

struct MLine {
    size_t qb, qs, tb, ts, mb, ms, vb, vs;
    wreal dinv;
    int n, open, spec;
};

static __device__ __forceinline__ int ml_cl(MLine L, int c) { return L.open ? max(c - 1, 0) : awrap(c - 1, L.n); }
static __device__ __forceinline__ int ml_cr(MLine L, int c) { return L.open ? min(c, L.n - 1) : awrap(c, L.n); }
static __device__ __forceinline__ bool ml_on(MLine L, int c) { return !(L.open && (c == 0 || c == L.n)); }
static __device__ __forceinline__ wreal ml_V(const wreal* vel, MLine L, int m)
{
    return (wreal)0.5 * (vel[L.vb + (size_t)m * L.vs] + vel[L.vb + (size_t)(m + 1) * L.vs]);
}
static __device__ __forceinline__ wreal ml_q(const wreal* q, MLine L, int c)
{
    if (!L.open) c = awrap(c, L.n);
    else if (c < 0 || c > L.n) return 0;
    return q[L.qb + (size_t)c * L.qs];
}
static __device__ int ml_type(const wreal* q, MLine L, int m)
{
    if (!L.open) return FT_FH;
    int d = min(m, L.n - 1 - m);
    if (d == 0) {
        if (L.spec) {
            if (m == 0) { if (ml_q(q, L, 1) < (wreal)0) return FT_Q0; }
            else        { if (ml_q(q, L, m) > (wreal)0) return FT_QM1; }
        }
        return FT_F2;
    }
    return d == 1 ? FT_F3H : FT_FH;
}
static __device__ wreal ml_aF_small(const wreal* at, const wreal* msf, MLine L, int m)
{
    wreal r = 0;
    for (int a = 0; a < 4; ++a) {
        int c = a == 0 ? m : a == 1 ? m + 1 : a == 2 ? 0 : L.n;
        if ((a == 1 && c > L.n) || (a == 2 && (m == 0 || m + 1 == 0))
            || (a == 3 && (m == L.n || m + 1 == L.n)))
            continue;
        if (!ml_on(L, c)) continue;
        wreal cf = msf[L.mb + (size_t)c * L.ms] * L.dinv;
        wreal tv = at[L.tb + (size_t)c * L.ts];
        if (ml_cl(L, c) == m) r += cf * tv;
        if (ml_cr(L, c) == m) r -= cf * tv;
    }
    return r;
}
static __device__ __forceinline__ wreal ml_term(const wreal* at, const wreal* msf, MLine L, int c)
{
    return ml_on(L, c) ? (msf[L.mb + (size_t)c * L.ms] * L.dinv) * at[L.tb + (size_t)c * L.ts]
                       : (wreal)0;
}
// face m's flux adjoint: + at the point above it (whose cl it is), - at the
// point below (whose cr it is); a periodic line's end points 0 and n take
// faces n-1 and 0 as well.  Lines under two cells take the general form.
static __device__ __forceinline__ wreal ml_aF(const wreal* at, const wreal* msf, MLine L, int m)
{
    if (L.n < 2) return ml_aF_small(at, msf, L, m);
    wreal r = ml_term(at, msf, L, m + 1) - ml_term(at, msf, L, m);
    if (!L.open) {
        if (m == L.n - 1) r += ml_term(at, msf, L, 0);
        if (m == 0) r -= ml_term(at, msf, L, L.n);
    }
    return r;
}
static __device__ __forceinline__ wreal ml_slot(int S, const wreal* q, const wreal* at, const wreal* msf,
                                                const wreal* vel, MLine L, int p)
{
    int m = L.open ? p + 2 - S : awrap(p + 2 - S, L.n);
    if (m < 0 || m > L.n - 1) return 0;
    return ml_aF(at, msf, L, m) * fw_qs(S, ml_type(q, L, m), ml_V(vel, L, m), 0, 0);
}
static __device__ wreal ml_aq(const wreal* q, const wreal* at, const wreal* msf,
                              const wreal* vel, MLine L, int p)
{
    if (!L.open && p == L.n) return 0;
    return ml_slot(0, q, at, msf, vel, L, p) + ml_slot(1, q, at, msf, vel, L, p)
         + ml_slot(2, q, at, msf, vel, L, p) + ml_slot(3, q, at, msf, vel, L, p)
         + ml_slot(4, q, at, msf, vel, L, p) + ml_slot(5, q, at, msf, vel, L, p);
}
static __device__ wreal ml_aV(const wreal* q, const wreal* at, const wreal* msf,
                              const wreal* vel, MLine L, int m)
{
    FW w = fw_make(ml_type(q, L, m), 0, 0);
    return ml_aF(at, msf, L, m)
         * fw_v(w, ml_V(vel, L, m), ml_q(q, L, m - 2), ml_q(q, L, m - 1), ml_q(q, L, m),
                ml_q(q, L, m + 1), ml_q(q, L, m + 2), ml_q(q, L, m + 3));
}
// adjoint into vel[f], f = 0..n
static __device__ wreal ml_avel(const wreal* q, const wreal* at, const wreal* msf,
                                const wreal* vel, MLine L, int f)
{
    wreal r = 0;
    if (f >= 1) r += (wreal)0.5 * ml_aV(q, at, msf, vel, L, f - 1);
    if (f <= L.n - 1) r += (wreal)0.5 * ml_aV(q, at, msf, vel, L, f);
    return r;
}

// ------------------------------------------- half-level vertical lines

struct ZLine {
    size_t qb, qs, tb, vb1, vb2, vs;
    wreal a1, a2;
    int nz, on, vorder;
};

static __device__ __forceinline__ wreal zl_V(const wreal* rw, ZLine L, int kf)
{
    return L.a1 * rw[L.vb1 + (size_t)kf * L.vs] + L.a2 * rw[L.vb2 + (size_t)kf * L.vs];
}
static __device__ int zl_type(ZLine L, int kf)
{
    int nz = L.nz, vo = L.vorder;
    if (kf == 0 || kf == nz) return FT_ZERO;
    if (vo == 2 || kf == 1 || kf == nz - 1) return FT_Z2;
    if (vo == 6 && kf >= 3 && kf <= nz - 3) return FT_F6;
    if (vo == 4 || vo == 6) return FT_F4;
    if (vo == 5 && kf >= 3 && kf <= nz - 3) return FT_F5V;
    return FT_F3V;
}
static __device__ __forceinline__ wreal zl_q(const wreal* q, ZLine L, int k)
{
    return (k < 0 || k >= L.nz) ? (wreal)0 : q[L.qb + (size_t)k * L.qs];
}
static __device__ wreal zl_aF(const wreal* at, const wreal* rdz, ZLine L, int kf)
{
    if (!L.on) return 0;
    wreal r = 0;
    if (kf < L.nz) r += rdz[kf] * at[L.tb + (size_t)kf * L.qs];
    if (kf >= 1) r -= rdz[kf - 1] * at[L.tb + (size_t)(kf - 1) * L.qs];
    return r;
}
static __device__ __forceinline__ wreal zl_slot(int S, const wreal* at, const wreal* rw, const wreal* rdz,
                                                const wreal* fnm, const wreal* fnp, ZLine L, int p)
{
    int kf = p + 3 - S;
    if (kf < 0 || kf > L.nz) return 0;
    int t = zl_type(L, kf);
    if (t == FT_ZERO) return 0;
    return zl_aF(at, rdz, L, kf) * fw_qs(S, t, zl_V(rw, L, kf), fnm[kf], fnp[kf]);
}
static __device__ wreal zl_aq(const wreal* at, const wreal* rw, const wreal* rdz,
                              const wreal* fnm, const wreal* fnp, ZLine L, int p)
{
    if (!L.on) return 0;
    return zl_slot(0, at, rw, rdz, fnm, fnp, L, p) + zl_slot(1, at, rw, rdz, fnm, fnp, L, p)
         + zl_slot(2, at, rw, rdz, fnm, fnp, L, p) + zl_slot(3, at, rw, rdz, fnm, fnp, L, p)
         + zl_slot(4, at, rw, rdz, fnm, fnp, L, p) + zl_slot(5, at, rw, rdz, fnm, fnp, L, p);
}
static __device__ wreal zl_aV(const wreal* q, const wreal* at, const wreal* rw, const wreal* rdz,
                              const wreal* fnm, const wreal* fnp, ZLine L, int kf)
{
    if (!L.on) return 0;
    int t = zl_type(L, kf);
    if (t == FT_ZERO) return 0;
    FW w = fw_make(t, fnm[kf], fnp[kf]);
    return zl_aF(at, rdz, L, kf)
         * fw_v(w, zl_V(rw, L, kf), zl_q(q, L, kf - 3), zl_q(q, L, kf - 2), zl_q(q, L, kf - 1),
                zl_q(q, L, kf), zl_q(q, L, kf + 1), zl_q(q, L, kf + 2));
}

// ----------------------------------------------------------- AD kernels

#define AD_ARGS                                                                   \
    const wreal* __restrict__ at, const wreal* __restrict__ q,                    \
    const wreal* __restrict__ ru, const wreal* __restrict__ rv,                   \
    const wreal* __restrict__ rw, wreal* __restrict__ aq,                         \
    wreal* __restrict__ aru, wreal* __restrict__ arv, wreal* __restrict__ arw,    \
    const wreal* __restrict__ msf, const wreal* __restrict__ rdz,                 \
    const wreal* __restrict__ fnm, const wreal* __restrict__ fnp,                 \
    wreal dx_inv, wreal dy_inv, int nz, int ny, int nx,                           \
    int open_x, int open_y, int spec, int vorder

#define AD_INDEX                                                                  \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                \
    int j = blockIdx.y * blockDim.y + threadIdx.y;                                \
    int k = blockIdx.z * blockDim.z + threadIdx.z;                                \
    if (i > nx || j > ny || k > nz) return;

static __device__ CLine cline(size_t qb, size_t qs, size_t tb, size_t ts, size_t mb, size_t ms,
                              size_t vb1, size_t vb2, size_t vs, wreal a1, wreal a2,
                              wreal dinv, int n, int open, int flav, int on)
{
    CLine L;
    L.qb = qb; L.qs = qs; L.tb = tb; L.ts = ts; L.mb = mb; L.ms = ms;
    L.vb1 = vb1; L.vb2 = vb2; L.vs = vs; L.a1 = a1; L.a2 = a2; L.dinv = dinv;
    L.n = n; L.open = open; L.flav = flav; L.on = on;
    return L;
}
static __device__ ZLine zline(size_t qb, size_t qs, size_t tb, size_t vb1, size_t vb2,
                              size_t vs, wreal a1, wreal a2, int nz, int on, int vorder)
{
    ZLine L;
    L.qb = qb; L.qs = qs; L.tb = tb; L.vb1 = vb1; L.vb2 = vb2; L.vs = vs;
    L.a1 = a1; L.a2 = a2; L.nz = nz; L.on = on; L.vorder = vorder;
    return L;
}
static __device__ MLine mline(size_t qb, size_t qs, size_t mb, size_t ms, size_t vb, size_t vs,
                              wreal dinv, int n, int open, int spec)
{
    MLine L;
    L.qb = qb; L.qs = qs; L.tb = qb; L.ts = qs; L.mb = mb; L.ms = ms;
    L.vb = vb; L.vs = vs; L.dinv = dinv; L.n = n; L.open = open; L.spec = spec;
    return L;
}

// ------------------------------------------------------------ scalar AD

static __device__ CLine sc_xline(int k, int j, int ny, int nx, wreal dx_inv, int open_x, int spec)
{
    size_t b = R3(k, j, 0, ny, nx);
    size_t v = R3(k, j, 0, ny, nx + 1);
    return cline(b, 1, b, 1, (size_t)j * nx, 1, v, v, 1, (wreal)1, (wreal)0, dx_inv, nx, open_x,
                 open_x && !spec ? 1 : 0, 1);
}
static __device__ CLine sc_yline(int k, int i, int ny, int nx, wreal dy_inv, int open_y, int spec)
{
    size_t b = R3(k, 0, i, ny, nx);
    size_t v = R3(k, 0, i, ny + 1, nx);
    return cline(b, nx, b, nx, i, nx, v, v, nx, (wreal)1, (wreal)0, dy_inv, ny, open_y,
                 open_y && !spec ? 1 : 0, 1);
}

static __device__ __forceinline__ void adv_scalar_ad_body(int part, AD_ARGS)
{
    AD_INDEX
    size_t plane = (size_t)ny * nx;
    if ((part == 0 || part >= 4) && k < nz && j < ny && i < nx) {
        wreal acc;
        if (part == 0) {
            acc = cl_aq(at, msf, ru, sc_xline(k, j, ny, nx, dx_inv, open_x, spec), i, spec);
        } else if (part == 4) {
            acc = cl_aq(at, msf, rv, sc_yline(k, i, ny, nx, dy_inv, open_y, spec), j, spec);
        } else {
            ZLine Z = zline(R3(0, j, i, ny, nx), plane, R3(0, j, i, ny, nx), R3(0, j, i, ny, nx),
                            R3(0, j, i, ny, nx), plane, (wreal)1, (wreal)0, nz, 1, vorder);
            acc = zl_aq(at, rw, rdz, fnm, fnp, Z, k);
        }
        aq[R3(k, j, i, ny, nx)] += acc;
    }
    if (part == 1 && k < nz && j < ny) {
        CLine X = sc_xline(k, j, ny, nx, dx_inv, open_x, spec);
        aru[R3(k, j, i, ny, nx + 1)] += cl_aV(q, at, msf, ru, X, i, spec);
    }
    if (part == 2 && k < nz && i < nx) {
        CLine Y = sc_yline(k, i, ny, nx, dy_inv, open_y, spec);
        arv[R3(k, j, i, ny + 1, nx)] += cl_aV(q, at, msf, rv, Y, j, spec);
    }
    if (part == 3 && j < ny && i < nx) {
        ZLine Z = zline(R3(0, j, i, ny, nx), plane, R3(0, j, i, ny, nx), R3(0, j, i, ny, nx),
                        R3(0, j, i, ny, nx), plane, (wreal)1, (wreal)0, nz, 1, vorder);
        arw[R3(k, j, i, ny, nx)] += zl_aV(q, at, rw, rdz, fnm, fnp, Z, k);
    }
}

// ----------------------------------------------------------------- u AD

static __device__ __forceinline__ bool u_xedge(int i, int nx, int open_x) { return open_x && (i == 0 || i == nx); }
static __device__ __forceinline__ int u_cl(int i, int nx, int open_x) { return open_x ? max(i - 1, 0) : awrap(i - 1, nx); }
static __device__ __forceinline__ int u_cr(int i, int nx, int open_x) { return open_x ? min(i, nx - 1) : awrap(i, nx); }
static __device__ __forceinline__ int u_iu(int i, int nx, int open_x) { return open_x ? i : awrap(i, nx); }

static __device__ CLine u_yline(int k, int i, int ny, int nx, wreal dy_inv, int open_x,
                                int open_y, int spec)
{
    int cl = u_cl(i, nx, open_x), cr = u_cr(i, nx, open_x), iu = u_iu(i, nx, open_x);
    return cline(R3(k, 0, iu, ny, nx + 1), nx + 1, R3(k, 0, i, ny, nx + 1), nx + 1, i, nx + 1,
                 R3(k, 0, cl, ny + 1, nx), R3(k, 0, cr, ny + 1, nx), nx, (wreal)0.5, (wreal)0.5,
                 dy_inv, ny, open_y, open_y && !spec ? 2 : 0, !u_xedge(i, nx, open_x));
}
static __device__ ZLine u_zline(int j, int i, int nz, int ny, int nx, int open_x, int open_y,
                                int vorder)
{
    int cl = u_cl(i, nx, open_x), cr = u_cr(i, nx, open_x), iu = u_iu(i, nx, open_x);
    size_t plane = (size_t)ny * (nx + 1), rplane = (size_t)ny * nx;
    size_t b1, b2;
    wreal a1, a2;
    if (open_x && i == 0) {
        b1 = b2 = R3(0, j, 0, ny, nx); a1 = 1; a2 = 0;
    } else if (open_x && i == nx) {
        b1 = b2 = R3(0, j, nx - 1, ny, nx); a1 = 1; a2 = 0;
    } else {
        b1 = R3(0, j, cl, ny, nx); b2 = R3(0, j, cr, ny, nx); a1 = (wreal)0.5; a2 = (wreal)0.5;
    }
    return zline(R3(0, j, iu, ny, nx + 1), plane, R3(0, j, i, ny, nx + 1), b1, b2, rplane, a1, a2,
                 nz, !(u_xedge(i, nx, open_x) && open_y), vorder);
}
// weight of mass column c in u-point i's corner-averaged flux
static __device__ __forceinline__ wreal u_wcol(int i, int c, int nx, int open_x)
{
    return (wreal)0.5 * ((u_cl(i, nx, open_x) == c ? 1 : 0) + (u_cr(i, nx, open_x) == c ? 1 : 0));
}
static __device__ __forceinline__ wreal u_wzcol(int i, int c, int nx, int open_x)
{
    if (open_x && i == 0) return c == 0 ? (wreal)1 : (wreal)0;
    if (open_x && i == nx) return c == nx - 1 ? (wreal)1 : (wreal)0;
    return u_wcol(i, c, nx, open_x);
}

static __device__ __forceinline__ void adv_u_ad_body(int part, AD_ARGS)
{
    AD_INDEX
    if ((part <= 1 || part >= 4) && k < nz && j < ny) {       // u points (k, j, i<=nx)
        MLine X = mline(R3(k, j, 0, ny, nx + 1), 1, (size_t)j * (nx + 1), 1,
                        R3(k, j, 0, ny, nx + 1), 1, dx_inv, nx, open_x, spec);
        if (part == 0) {
            aq[R3(k, j, i, ny, nx + 1)] += ml_aq(q, at, msf, ru, X, i);
        } else if (part >= 4) {
        wreal acc = 0;
        // u-points whose y (part 4) or z (part 5) lines read column i
        int nc = (!open_x && i == 0 && nx != 0) ? 2 : 1;
        for (int a = 0; a < nc; ++a) {
            int iu = a == 0 ? i : nx;
            if (u_iu(iu, nx, open_x) != i) continue;
            if (part == 4)
                acc += cl_aq(at, msf, rv, u_yline(k, iu, ny, nx, dy_inv, open_x, open_y, spec),
                             j, spec);
            else
                acc += zl_aq(at, rw, rdz, fnm, fnp,
                             u_zline(j, iu, nz, ny, nx, open_x, open_y, vorder), k);
        }
        aq[R3(k, j, i, ny, nx + 1)] += acc;
        } else {
        aru[R3(k, j, i, ny, nx + 1)] += ml_avel(q, at, msf, ru, X, i);
        }
    }
    if (part == 2 && k < nz && i < nx) {                      // rv (k, j<=ny, i<nx)
        wreal acc = 0;
        for (int a = 0; a < 4; ++a) {
            int c = a == 0 ? i : a == 1 ? i + 1 : a == 2 ? 0 : nx;
            bool dup = (a >= 1 && c == i) || (a >= 2 && c == i + 1) || (a == 3 && c == 0);
            if (dup || c > nx) continue;
            wreal wt = u_wcol(c, i, nx, open_x);
            if (wt == (wreal)0) continue;
            CLine Y = u_yline(k, c, ny, nx, dy_inv, open_x, open_y, spec);
            acc += wt * cl_aV(q, at, msf, rv, Y, j, spec);
        }
        arv[R3(k, j, i, ny + 1, nx)] += acc;
    }
    if (part == 3 && j < ny && i < nx) {                      // rw (k<=nz, j, i<nx)
        wreal acc = 0;
        for (int a = 0; a < 4; ++a) {
            int c = a == 0 ? i : a == 1 ? i + 1 : a == 2 ? 0 : nx;
            bool dup = (a >= 1 && c == i) || (a >= 2 && c == i + 1) || (a == 3 && c == 0);
            if (dup || c > nx) continue;
            wreal wt = u_wzcol(c, i, nx, open_x);
            if (wt == (wreal)0) continue;
            ZLine Z = u_zline(j, c, nz, ny, nx, open_x, open_y, vorder);
            acc += wt * zl_aV(q, at, rw, rdz, fnm, fnp, Z, k);
        }
        arw[R3(k, j, i, ny, nx)] += acc;
    }
}

// ----------------------------------------------------------------- v AD

static __device__ __forceinline__ bool v_yedge(int j, int ny, int open_y) { return open_y && (j == 0 || j == ny); }
static __device__ __forceinline__ int v_rs(int j, int ny, int open_y) { return open_y ? max(j - 1, 0) : awrap(j - 1, ny); }
static __device__ __forceinline__ int v_rn(int j, int ny, int open_y) { return open_y ? min(j, ny - 1) : awrap(j, ny); }
static __device__ __forceinline__ int v_jv(int j, int ny, int open_y) { return open_y ? j : awrap(j, ny); }

static __device__ CLine v_xline(int k, int j, int ny, int nx, wreal dx_inv, int open_x,
                                int open_y, int spec)
{
    int rs = v_rs(j, ny, open_y), rn = v_rn(j, ny, open_y), jv = v_jv(j, ny, open_y);
    return cline(R3(k, jv, 0, ny + 1, nx), 1, R3(k, j, 0, ny + 1, nx), 1, (size_t)j * nx, 1,
                 R3(k, rs, 0, ny, nx + 1), R3(k, rn, 0, ny, nx + 1), 1, (wreal)0.5, (wreal)0.5,
                 dx_inv, nx, open_x, open_x && !spec ? 2 : 0, !v_yedge(j, ny, open_y));
}
static __device__ ZLine v_zline(int j, int i, int nz, int ny, int nx, int open_y, int vorder)
{
    int rs = v_rs(j, ny, open_y), rn = v_rn(j, ny, open_y), jv = v_jv(j, ny, open_y);
    return zline(R3(0, jv, i, ny + 1, nx), (size_t)(ny + 1) * nx, R3(0, j, i, ny + 1, nx),
                 R3(0, rs, i, ny, nx), R3(0, rn, i, ny, nx), (size_t)ny * nx, (wreal)0.5,
                 (wreal)0.5, nz, !v_yedge(j, ny, open_y), vorder);
}
static __device__ __forceinline__ wreal v_wrow(int j, int r, int ny, int open_y)
{
    return (wreal)0.5 * ((v_rs(j, ny, open_y) == r ? 1 : 0) + (v_rn(j, ny, open_y) == r ? 1 : 0));
}

static __device__ __forceinline__ void adv_v_ad_body(int part, AD_ARGS)
{
    AD_INDEX
    if ((part == 0 || part == 2 || part >= 4) && k < nz && i < nx) { // v points (k, j<=ny, i)
        MLine Y = mline(R3(k, 0, i, ny + 1, nx), nx, i, nx, R3(k, 0, i, ny + 1, nx), nx, dy_inv,
                        ny, open_y, spec);
        if (part == 0) {
            aq[R3(k, j, i, ny + 1, nx)] += ml_aq(q, at, msf, rv, Y, j);
        } else if (part >= 4) {
        wreal acc = 0;
        // v-points whose x (part 4) or z (part 5) lines read row j
        int nc = (!open_y && j == 0 && ny != 0) ? 2 : 1;
        for (int a = 0; a < nc; ++a) {
            int jj = a == 0 ? j : ny;
            if (v_jv(jj, ny, open_y) != j) continue;
            if (part == 4)
                acc += cl_aq(at, msf, ru, v_xline(k, jj, ny, nx, dx_inv, open_x, open_y, spec),
                             i, spec);
            else
                acc += zl_aq(at, rw, rdz, fnm, fnp, v_zline(jj, i, nz, ny, nx, open_y, vorder), k);
        }
        aq[R3(k, j, i, ny + 1, nx)] += acc;
        } else {
        arv[R3(k, j, i, ny + 1, nx)] += ml_avel(q, at, msf, rv, Y, j);
        }
    }
    if (part == 1 && k < nz && j < ny) {                      // ru (k, j<ny, i<=nx)
        wreal acc = 0;
        for (int a = 0; a < 4; ++a) {
            int c = a == 0 ? j : a == 1 ? j + 1 : a == 2 ? 0 : ny;
            bool dup = (a >= 1 && c == j) || (a >= 2 && c == j + 1) || (a == 3 && c == 0);
            if (dup || c > ny) continue;
            wreal wt = v_wrow(c, j, ny, open_y);
            if (wt == (wreal)0) continue;
            CLine X = v_xline(k, c, ny, nx, dx_inv, open_x, open_y, spec);
            acc += wt * cl_aV(q, at, msf, ru, X, i, spec);
        }
        aru[R3(k, j, i, ny, nx + 1)] += acc;
    }
    if (part == 3 && j < ny && i < nx) {                      // rw (k<=nz, j<ny, i)
        wreal acc = 0;
        for (int a = 0; a < 4; ++a) {
            int c = a == 0 ? j : a == 1 ? j + 1 : a == 2 ? 0 : ny;
            bool dup = (a >= 1 && c == j) || (a >= 2 && c == j + 1) || (a == 3 && c == 0);
            if (dup || c > ny) continue;
            wreal wt = v_wrow(c, j, ny, open_y);
            if (wt == (wreal)0) continue;
            ZLine Z = v_zline(c, i, nz, ny, nx, open_y, vorder);
            acc += wt * zl_aV(q, at, rw, rdz, fnm, fnp, Z, k);
        }
        arw[R3(k, j, i, ny, nx)] += acc;
    }
}

// ----------------------------------------------------------------- w AD
//
// Tendency levels k = 1..nz-1 and the lid k = nz (nz >= 2).  Level k's x
// and y face fluxes interpolate ru/rv from levels k and k-1 (fnm, fnp);
// the lid's extrapolate from nz-1 and nz-2 and carry no non-cb terms.
// The vertical faces sit on mass levels m = 0..nz-1 (q0 point m+1).

static __device__ __forceinline__ bool w_level(int k, int nz) { return (k >= 1 && k <= nz - 1) || (k == nz && nz >= 2); }

static __device__ CLine w_xline(int k, int j, int nz, int ny, int nx, wreal dx_inv,
                                const wreal* fnm, const wreal* fnp, int open_x, int spec)
{
    size_t b = R3(k, j, 0, ny, nx);
    bool lid = k == nz;
    int k1 = lid ? nz - 1 : k, k2 = lid ? nz - 2 : k - 1;
    wreal a1 = lid ? (wreal)2 - fnm[nz - 1] : fnm[k];
    wreal a2 = lid ? -fnp[nz - 1] : fnp[k];
    return cline(b, 1, b, 1, (size_t)j * nx, 1, R3(k1, j, 0, ny, nx + 1), R3(k2, j, 0, ny, nx + 1), 1,
                 a1, a2, dx_inv, nx, open_x, (open_x && !spec && !lid) ? 1 : 0, w_level(k, nz));
}
static __device__ CLine w_yline(int k, int i, int nz, int ny, int nx, wreal dy_inv,
                                const wreal* fnm, const wreal* fnp, int open_y, int spec)
{
    size_t b = R3(k, 0, i, ny, nx);
    bool lid = k == nz;
    int k1 = lid ? nz - 1 : k, k2 = lid ? nz - 2 : k - 1;
    wreal a1 = lid ? (wreal)2 - fnm[nz - 1] : fnm[k];
    wreal a2 = lid ? -fnp[nz - 1] : fnp[k];
    return cline(b, nx, b, nx, i, nx, R3(k1, 0, i, ny + 1, nx), R3(k2, 0, i, ny + 1, nx), nx,
                 a1, a2, dy_inv, ny, open_y, (open_y && !spec && !lid) ? 1 : 0, w_level(k, nz));
}
static __device__ int wz_type(int m, int nz, int vorder)
{
    if (vorder == 2 || vorder == 4 || vorder == 6) {
        if (vorder == 2 || m == 0 || m == nz - 1) return FT_F2;
        if (vorder == 6 && m >= 2 && m <= nz - 3) return FT_F6;
        return FT_F4;
    }
    if (m == 0 || m == nz - 1) return FT_F2;
    if (vorder == 5 && m >= 2 && m <= nz - 3) return FT_F5V;
    return FT_F3V;
}
static __device__ wreal wz_aF(const wreal* at, const wreal* rdz, size_t col, size_t plane,
                              int m, int nz)
{
    wreal r = 0;
    if (m >= 1 && m <= nz - 1) r -= rdz[m] * at[col + (size_t)m * plane];
    if (m + 1 <= nz - 1) r += rdz[m + 1] * at[col + (size_t)(m + 1) * plane];
    if (m == nz - 1 && nz >= 2) r += ((wreal)2 * rdz[nz - 1]) * at[col + (size_t)nz * plane];
    return r;
}
static __device__ __forceinline__ wreal wz_V(const wreal* rw, size_t col, size_t plane, int m)
{
    return (wreal)0.5 * (rw[col + (size_t)m * plane] + rw[col + (size_t)(m + 1) * plane]);
}
static __device__ __forceinline__ wreal wz_q(const wreal* q, size_t col, size_t plane, int l, int nz)
{
    return (l < 0 || l > nz) ? (wreal)0 : q[col + (size_t)l * plane];
}
static __device__ wreal wz_aV(const wreal* q, const wreal* at, const wreal* rw, const wreal* rdz,
                              size_t col, size_t plane, int m, int nz, int vorder)
{
    FW w = fw_make(wz_type(m, nz, vorder), 0, 0);
    return wz_aF(at, rdz, col, plane, m, nz)
         * fw_v(w, wz_V(rw, col, plane, m), wz_q(q, col, plane, m - 2, nz),
                wz_q(q, col, plane, m - 1, nz), wz_q(q, col, plane, m, nz),
                wz_q(q, col, plane, m + 1, nz), wz_q(q, col, plane, m + 2, nz),
                wz_q(q, col, plane, m + 3, nz));
}

static __device__ __forceinline__ wreal wz_slot(int S, const wreal* at, const wreal* rw, const wreal* rdz,
                                                size_t col, size_t plane, int k, int nz, int vorder)
{
    int m = k + 2 - S;
    if (m < 0 || m > nz - 1) return 0;
    return wz_aF(at, rdz, col, plane, m, nz)
         * fw_qs(S, wz_type(m, nz, vorder), wz_V(rw, col, plane, m), 0, 0);
}

static __device__ __forceinline__ void adv_w_ad_body(int part, AD_ARGS)
{
    AD_INDEX
    size_t plane = (size_t)ny * nx;
    if ((part == 0 || part == 3 || part >= 4) && j < ny && i < nx) { // w points, rw (k<=nz)
        size_t col = R3(0, j, i, ny, nx);
        if (part != 3) {
        wreal acc = 0;
        if (part == 0 && w_level(k, nz))
            acc = cl_aq(at, msf, ru, w_xline(k, j, nz, ny, nx, dx_inv, fnm, fnp, open_x, spec),
                        i, spec);
        if (part == 4 && w_level(k, nz))
            acc = cl_aq(at, msf, rv, w_yline(k, i, nz, ny, nx, dy_inv, fnm, fnp, open_y, spec),
                        j, spec);
        if (part == 5 && nz >= 2) {
            acc += wz_slot(0, at, rw, rdz, col, plane, k, nz, vorder)
                 + wz_slot(1, at, rw, rdz, col, plane, k, nz, vorder)
                 + wz_slot(2, at, rw, rdz, col, plane, k, nz, vorder)
                 + wz_slot(3, at, rw, rdz, col, plane, k, nz, vorder)
                 + wz_slot(4, at, rw, rdz, col, plane, k, nz, vorder)
                 + wz_slot(5, at, rw, rdz, col, plane, k, nz, vorder);
        }
        aq[R3(k, j, i, ny, nx)] += acc;
        } else {
        wreal ar = 0;
        if (nz >= 2) {
            if (k >= 1) ar += (wreal)0.5 * wz_aV(q, at, rw, rdz, col, plane, k - 1, nz, vorder);
            if (k <= nz - 1) ar += (wreal)0.5 * wz_aV(q, at, rw, rdz, col, plane, k, nz, vorder);
        }
        arw[R3(k, j, i, ny, nx)] += ar;
        }
    }
    if (part == 1 && k < nz && j < ny) {                      // ru (k<nz, j, i<=nx)
        wreal acc = 0;
        // level k (fnm), level k+1 (fnp), the lid (2 - fnm / -fnp)
        int lv[3] = {k, k + 1, nz};
        for (int a = 0; a < 3; ++a) {
            int l = lv[a];
            if (!w_level(l, nz)) continue;
            if (a == 1 && l == nz) continue;                  // the lid comes once, below
            if (a == 0 && l == nz) continue;
            wreal wt;
            if (l == nz) wt = k == nz - 1 ? (wreal)2 - fnm[nz - 1] : k == nz - 2 ? -fnp[nz - 1] : (wreal)0;
            else wt = a == 0 ? fnm[l] : fnp[l];
            if (wt == (wreal)0) continue;
            CLine X = w_xline(l, j, nz, ny, nx, dx_inv, fnm, fnp, open_x, spec);
            acc += wt * cl_aV(q, at, msf, ru, X, i, spec);
        }
        aru[R3(k, j, i, ny, nx + 1)] += acc;
    }
    if (part == 2 && k < nz && i < nx) {                      // rv (k<nz, j<=ny, i)
        wreal acc = 0;
        int lv[3] = {k, k + 1, nz};
        for (int a = 0; a < 3; ++a) {
            int l = lv[a];
            if (!w_level(l, nz)) continue;
            if (a == 1 && l == nz) continue;
            if (a == 0 && l == nz) continue;
            wreal wt;
            if (l == nz) wt = k == nz - 1 ? (wreal)2 - fnm[nz - 1] : k == nz - 2 ? -fnp[nz - 1] : (wreal)0;
            else wt = a == 0 ? fnm[l] : fnp[l];
            if (wt == (wreal)0) continue;
            CLine Y = w_yline(l, i, nz, ny, nx, dy_inv, fnm, fnp, open_y, spec);
            acc += wt * cl_aV(q, at, msf, rv, Y, j, spec);
        }
        arv[R3(k, j, i, ny + 1, nx)] += acc;
    }
}

// ------------------------------------------------- AD entry points
//
// One kernel per adjoint output (ru, rv, rw) and per direction of the
// field adjoint (q: x, y, z, three passes adding into aq): the gathers
// share no work worth the registers of one fused kernel.

#define AD_PASS at, q, ru, rv, rw, aq, aru, arv, arw, msf, rdz, fnm, fnp, dx_inv, dy_inv, \
    nz, ny, nx, open_x, open_y, spec, vorder
extern "C" __global__ void adv_scalar_ad_qx(AD_ARGS) { adv_scalar_ad_body(0, AD_PASS); }
extern "C" __global__ void adv_scalar_ad_qy(AD_ARGS) { adv_scalar_ad_body(4, AD_PASS); }
extern "C" __global__ void adv_scalar_ad_qz(AD_ARGS) { adv_scalar_ad_body(5, AD_PASS); }
extern "C" __global__ void adv_scalar_ad_ru(AD_ARGS) { adv_scalar_ad_body(1, AD_PASS); }
extern "C" __global__ void adv_scalar_ad_rv(AD_ARGS) { adv_scalar_ad_body(2, AD_PASS); }
extern "C" __global__ void adv_scalar_ad_rw(AD_ARGS) { adv_scalar_ad_body(3, AD_PASS); }
extern "C" __global__ void adv_u_ad_qx(AD_ARGS) { adv_u_ad_body(0, AD_PASS); }
extern "C" __global__ void adv_u_ad_qy(AD_ARGS) { adv_u_ad_body(4, AD_PASS); }
extern "C" __global__ void adv_u_ad_qz(AD_ARGS) { adv_u_ad_body(5, AD_PASS); }
extern "C" __global__ void adv_u_ad_ru(AD_ARGS) { adv_u_ad_body(1, AD_PASS); }
extern "C" __global__ void adv_u_ad_rv(AD_ARGS) { adv_u_ad_body(2, AD_PASS); }
extern "C" __global__ void adv_u_ad_rw(AD_ARGS) { adv_u_ad_body(3, AD_PASS); }
extern "C" __global__ void adv_v_ad_qx(AD_ARGS) { adv_v_ad_body(0, AD_PASS); }
extern "C" __global__ void adv_v_ad_qy(AD_ARGS) { adv_v_ad_body(4, AD_PASS); }
extern "C" __global__ void adv_v_ad_qz(AD_ARGS) { adv_v_ad_body(5, AD_PASS); }
extern "C" __global__ void adv_v_ad_ru(AD_ARGS) { adv_v_ad_body(1, AD_PASS); }
extern "C" __global__ void adv_v_ad_rv(AD_ARGS) { adv_v_ad_body(2, AD_PASS); }
extern "C" __global__ void adv_v_ad_rw(AD_ARGS) { adv_v_ad_body(3, AD_PASS); }
extern "C" __global__ void adv_w_ad_qx(AD_ARGS) { adv_w_ad_body(0, AD_PASS); }
extern "C" __global__ void adv_w_ad_qy(AD_ARGS) { adv_w_ad_body(4, AD_PASS); }
extern "C" __global__ void adv_w_ad_qz(AD_ARGS) { adv_w_ad_body(5, AD_PASS); }
extern "C" __global__ void adv_w_ad_ru(AD_ARGS) { adv_w_ad_body(1, AD_PASS); }
extern "C" __global__ void adv_w_ad_rv(AD_ARGS) { adv_w_ad_body(2, AD_PASS); }
extern "C" __global__ void adv_w_ad_rw(AD_ARGS) { adv_w_ad_body(3, AD_PASS); }
