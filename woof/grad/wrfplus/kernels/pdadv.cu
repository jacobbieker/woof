// Tangent-linear and adjoint of woof's strict positive-definite scalar
// transport (moist_adv_opt = 1; WRF module_advect_em.F advect_scalar_pd,
// the RK3 final stage): woof/core/kernels/pd_advection.cu's pd_fluxes and
// pd_renorm_apply, the GPUWM_WRF_EXACT_C_ADVECTION statements, periodic or
// specified lateral boundaries, map factors, h_sca_adv_order 3-6 (ADV_H)
// and v_sca_adv_order 2, 3, 4 and 6 (vorder).  The oracle is WRFPLUS 4.7.1
// G_/A_ADVECT_SCALAR_PD and WRF 4.7.1's advect_scalar_pd; see
// docs/dev/P3A-WRF-DRYCORE-ADJOINT.md.
//
// wreal is binary64 unless the unit is built with -DWRFPLUS_REAL=float.  The
// TL kernels are the primal statements on a value/tangent pair (Dl): the
// value half repeats the primal's operations in its order, so the float
// instantiation's base state gives the primal's words (the wrapper passes
// --fmad=false; divisions are __fdiv_rn).
//
// The limiter is linearised as WRFPLUS does: every branch (the Courant
// clamps of flux_upwind, |cr|, the max(0, .)/min(0, .) of flux_out, the
// limiter test flux_out > ph_low, max(0, ratio) and the donor chosen by the
// correction flux's sign) follows the base state, and a branch's constant
// arm (1, -1, 0) has no tangent.
//
// Three TL kernels, as the primal's two with the cell scale stored once:
//   pd_flux_tl   per face: upwind flux fl of the folded time-t scalar q0
//                and correction fc = fh(q) - fl;
//   pd_scale_tl  per cell: the renormalisation factor (1 where the
//                limiter does not act);
//   pd_apply_tl  per cell: tend = -div(fl + s*fc), s the donor's factor.
// The adjoint runs them backwards in gather form (one writer per word, no
// atomics): pd_apply_ad (faces), pd_scale_ad (cells), pd_flux_ad_face
// (faces: the limiter's share and the mass fluxes) and pd_flux_ad_cell
// (cells: q, q0 and the column mass by level).  A face flux's partial with
// respect to one input is its TL evaluated with a unit tangent on that
// input, so the adjoint is the TL's transpose by construction.
//
// Layout (k, j, i): cells (nz, ny, nx), x faces (nz, ny, nx+1), y faces
// (nz, ny+1, nx), eta faces (nz+1, ny, nx); columns (ny, nx).

#ifndef WRFPLUS_REAL
#define WRFPLUS_REAL double
#endif

// WRFPLUS_TL 0 builds the base-only forward (model.step_base): every
// tangent store is skipped and the compiler drops the tangent arithmetic.
#ifndef WRFPLUS_TL
#define WRFPLUS_TL 1
#endif
typedef WRFPLUS_REAL wreal;

#ifndef ADV_H
#define ADV_H 5
#endif

#define R3(k, j, i, nyy, nxx) \
    ((size_t)(k) * (size_t)(nyy) * (size_t)(nxx) + (size_t)(j) * (size_t)(nxx) + (size_t)(i))

static __device__ __forceinline__ int pwrap(int i, int n)
{
    int r = i < 0 ? i + n : (i >= n ? i - n : i);
    if ((unsigned)r >= (unsigned)n) {
        r = i % n;
        r = r < 0 ? r + n : r;
    }
    return r;
}

static __device__ __forceinline__ float wdiv(float a, float b) { return __fdiv_rn(a, b); }
static __device__ __forceinline__ double wdiv(double a, double b) { return a / b; }
static __device__ __forceinline__ float wsign(float v) { return copysignf(1.0f, v); }
static __device__ __forceinline__ double wsign(double v) { return copysign(1.0, v); }
static __device__ __forceinline__ float wabs(float v) { return fabsf(v); }
static __device__ __forceinline__ double wabs(double v) { return fabs(v); }
// the primal's argument order: fminf(1.0f, x), fmaxf(-1.0f, x), fmaxf(0.0f, x), ...
static __device__ __forceinline__ float wfmin(float a, float b) { return fminf(a, b); }
static __device__ __forceinline__ double wfmin(double a, double b) { return fmin(a, b); }
static __device__ __forceinline__ float wfmax(float a, float b) { return fmaxf(a, b); }
static __device__ __forceinline__ double wfmax(double a, double b) { return fmax(a, b); }

// pd_advection.cu's float literals; the binary64 build takes WRFPLUS's
// real*8 quotients.
#define KC(f32, num, den) (sizeof(wreal) == 4 ? (wreal)(f32) : (wreal)(num) / (wreal)(den))
#define C37_60 KC(0.6166666746139526f, 37.0, 60.0)
#define C2_15 KC(0.13333334028720856f, 2.0, 15.0)
#define C1_60 KC(0.01666666753590107f, 1.0, 60.0)
#define C7_12 KC(0.5833333134651184f, 7.0, 12.0)
#define C1_12 KC(0.0833333358168602f, 1.0, 12.0)
#define PD_EPS ((wreal)1e-20)

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
static __device__ __forceinline__ Dl operator+(Dl a, wreal b) { return dl(a.v + b, a.d); }
static __device__ __forceinline__ Dl operator+(wreal a, Dl b) { return dl(a + b.v, b.d); }
static __device__ __forceinline__ Dl operator*(Dl a, Dl b)
{
    return dl(a.v * b.v, a.d * b.v + a.v * b.d);
}
static __device__ __forceinline__ Dl operator*(wreal a, Dl b) { return dl(a * b.v, a * b.d); }
static __device__ __forceinline__ Dl operator*(Dl a, wreal b) { return dl(a.v * b, a.d * b); }
// a / c, c constant
static __device__ __forceinline__ Dl ddivc(Dl a, wreal c) { return dl(wdiv(a.v, c), a.d / c); }
// a / b
static __device__ __forceinline__ Dl ddiv(Dl a, Dl b)
{
    wreal r = wdiv(a.v, b.v);
    return dl(r, (a.d - r * b.d) / b.v);
}
// |a| (WRFPLUS: cr >= 0 takes +crd)
static __device__ __forceinline__ Dl dabs(Dl a) { return dl(wabs(a.v), a.v >= (wreal)0 ? a.d : -a.d); }
// fminf(1, a), fmaxf(-1, a), fmaxf(0, a), fminf(0, a): the base arm's tangent
static __device__ __forceinline__ Dl dmin1(Dl a)
{
    return dl(wfmin((wreal)1, a.v), (wreal)1 > a.v ? a.d : (wreal)0);
}
static __device__ __forceinline__ Dl dmaxm1(Dl a)
{
    return dl(wfmax((wreal)-1, a.v), (wreal)-1 < a.v ? a.d : (wreal)0);
}
static __device__ __forceinline__ Dl dmax0(Dl a)
{
    return dl(wfmax((wreal)0, a.v), (wreal)0 < a.v ? a.d : (wreal)0);
}
static __device__ __forceinline__ Dl dmin0(Dl a)
{
    return dl(wfmin((wreal)0, a.v), (wreal)0 > a.v ? a.d : (wreal)0);
}
static __device__ __forceinline__ Dl cst(wreal v) { return dl(v, (wreal)0); }

// a face's six-point stencil, by value (no pointers to thread-local arrays)
struct S6 { Dl s[6]; };

// ------------------------------------------------- the primal's face fluxes

// pd_flux5 for the unit's ADV_H (the exact build's statements)
static __device__ __forceinline__ Dl hflux(Dl qm3, Dl qm2, Dl qm1, Dl q0, Dl qp1, Dl qp2, Dl vel)
{
#if ADV_H == 2
    return ((wreal)0.5 * vel) * (q0 + qm1);
#elif ADV_H == 3
    Dl center = C7_12 * (q0 + qm1) - C1_12 * (qp1 + qm2);
    Dl dissipation = (wsign(vel.v) * C1_12) * ((qp1 - qm2) - (wreal)3 * (q0 - qm1));
    return vel * (center + dissipation);
#elif ADV_H == 4
    return vel * (C7_12 * (q0 + qm1) - C1_12 * (qp1 + qm2));
#elif ADV_H == 6
    return vel * (C37_60 * (q0 + qm1) - C2_15 * (qp1 + qm2) + C1_60 * (qp2 + qm3));
#else
    Dl center = C37_60 * (q0 + qm1) - C2_15 * (qp1 + qm2) + C1_60 * (qp2 + qm3);
    Dl dissipation = (wsign(vel.v) * C1_60)
        * ((qp2 - qm3) - (wreal)5 * (qp1 - qm2) + (wreal)10 * (q0 - qm1));
    return vel * (center - dissipation);
#endif
}

// pd_flux3h: horizontal third order two faces in from a specified boundary
static __device__ __forceinline__ Dl hflux3(Dl qm2, Dl qm1, Dl q0, Dl qp1, Dl vel)
{
#if ADV_H == 4 || ADV_H == 6
    return vel * (C7_12 * (q0 + qm1) - C1_12 * (qp1 + qm2));
#else
    Dl center = C7_12 * (q0 + qm1) - C1_12 * (qp1 + qm2);
    Dl dissipation = (wsign(vel.v) * C1_12) * ((qp1 - qm2) - (wreal)3 * (q0 - qm1));
    return vel * (center + dissipation);
#endif
}

// A horizontal face's high-order flux from its stencil s[0..5] = the
// field at face-3 .. face+2; band 0 full order, 1 second order, 2 flux3h.
static __device__ __forceinline__ Dl hface_high(S6 t, Dl vel, int band)
{
    if (band == 1 || ADV_H == 2) return ((wreal)0.5 * vel) * (t.s[3] + t.s[2]);
    if (band == 2) return hflux3(t.s[1], t.s[2], t.s[3], t.s[4], vel);
    return hflux(t.s[0], t.s[1], t.s[2], t.s[3], t.s[4], t.s[5], vel);
}

// flux_upwind's Courant arms times the face mass and spacing:
// muf*(dxp/dt) * (0.5*fminf(1, cr+|cr|)*qm1 + 0.5*fmaxf(-1, cr-|cr|)*q0v)
static __device__ __forceinline__ Dl upwind(Dl muf, wreal sp_dt, Dl qm1, Dl q0v, Dl cr)
{
    Dl a = dabs(cr);
    Dl fu = ((wreal)0.5 * dmin1(cr + a)) * qm1 + ((wreal)0.5 * dmaxm1(cr - a)) * q0v;
    return (muf * sp_dt) * fu;
}

// A horizontal face's upwind flux: right/left column masses and map
// factors, the cell spacing dx, the stage dt; ``west`` the western
// second-order face of a bounded x axis (WRF divides ru by the face mass
// before the Courant number there; pd_west_wrf).
static __device__ __forceinline__ Dl hface_low(Dl vel, Dl mr, Dl ml, wreal c1, wreal c2,
                                              wreal msr, wreal msl, int has_msf, wreal dx,
                                              wreal dt, Dl q0l, Dl q0r, int west)
{
    Dl muf = (wreal)0.5 * ((c1 * mr + c2) + (c1 * ml + c2));
    wreal dxp = dx;
    if (has_msf) dxp = wdiv(wdiv((wreal)2, msr + msl), wdiv((wreal)1, dx));
    Dl cr = ddiv(ddivc(vel * dt, dxp), muf);
    if (west) cr = ddivc(ddiv(vel, muf) * dt, dxp);
    return upwind(muf, wdiv(dxp, dt), q0l, q0r, cr);
}

// Does a bounded x axis's west face (i == 1) take WRF's (ru/mu)*dt/dx?
// open_x bit 2 is RunConfig.pd_adv_west_courant (finding FQ13): every
// other face's order there, as the primal's pd_fluxes; second order
// everywhere (ADV_H 2) has no such face in WRF.
static __device__ __forceinline__ int pd_west_wrf(int open_x)
{
    return open_x && !(open_x & 2) && ADV_H != 2;
}

// pd_zface_half from s[0..5] = q at kf-3 .. kf+2 (the face's interior
// levels only are read)
static __device__ __forceinline__ Dl zface_high(S6 t, Dl vel, int kf, int nz,
                                               wreal fnm, wreal fnp, int vorder)
{
    if (vorder == 2 || kf == 1 || kf == nz - 1)
        return vel * (fnm * t.s[3] + fnp * t.s[2]);
    if (vorder == 6 && kf >= 3 && kf <= nz - 3)
        return vel * (C37_60 * (t.s[3] + t.s[2]) - C2_15 * (t.s[4] + t.s[1]) + C1_60 * (t.s[5] + t.s[0]));
    if (vorder == 4 || vorder == 6)
        return vel * (C7_12 * (t.s[3] + t.s[2]) - C1_12 * (t.s[4] + t.s[1]));
    // vorder 3: pd_flux3 (Omega-signed dissipation)
    Dl center = C7_12 * (t.s[3] + t.s[2]) - C1_12 * (t.s[4] + t.s[1]);
    Dl dissipation = (wsign(vel.v) * C1_12) * ((t.s[4] - t.s[1]) - (wreal)3 * (t.s[3] - t.s[2]));
    return vel * (center - dissipation);
}

static __device__ __forceinline__ Dl zface_low(Dl vel, Dl mut, wreal c1, wreal c2, wreal rdk,
                                              wreal rdkm1, wreal dt, Dl q0l, Dl q0u)
{
    wreal dz = wdiv((wreal)2, rdk + rdkm1);
    Dl muf = c1 * mut + c2;
    Dl cr = ddiv(ddivc(vel * dt, dz), muf);
    return upwind(muf, wdiv(dz, dt), q0l, q0u, cr);
}

// ------------------------------------------------- face indexing helpers

// Horizontal face f of an axis of n cells: the stencil cells f-3+o
// (wrapped when periodic), the band and whether the face carries flux.
struct HFace { int cell[6]; int l; int r; int band; int live; };

static __device__ __forceinline__ HFace hface(int f, int n, int open)
{
    HFace h;
    if (open) {
        h.live = f > 0 && f < n;
        int d = min(f, n - f);
        h.band = d == 1 ? 1 : (d == 2 ? 2 : 0);
        for (int o = 0; o < 6; ++o) {
            int c = f - 3 + o;
            h.cell[o] = (c >= 0 && c < n) ? c : -1;
        }
        h.l = f - 1;
        h.r = f;
    } else {
        h.live = 1;
        h.band = 0;
        for (int o = 0; o < 6; ++o) h.cell[o] = pwrap(f - 3 + o, n);
        h.l = pwrap(f - 1, n);
        h.r = pwrap(f, n);
    }
    return h;
}

// which stencil entries the band reads
static __device__ __forceinline__ int hread(int band, int o)
{
    if (band == 1 || ADV_H == 2) return o == 2 || o == 3;
    if (band == 2) return o >= 1 && o <= 4;
    return 1;
}

// eta face kf's stencil levels kf-3 .. kf+2 that zface_high reads
static __device__ __forceinline__ int zread(int kf, int nz, int vorder, int o)
{
    int lev = kf - 3 + o;
    if (lev < 0 || lev >= nz) return 0;
    if (vorder == 2 || kf == 1 || kf == nz - 1) return o == 2 || o == 3;
    if (vorder == 6 && kf >= 3 && kf <= nz - 3) return 1;
    return o >= 1 && o <= 4;
}

// ----------------------------------------------------------------- TL

#define FLUX_TL_ARGS                                                               \
    const wreal* __restrict__ q, const wreal* __restrict__ qd,                     \
    const wreal* __restrict__ q0, const wreal* __restrict__ q0d,                   \
    const wreal* __restrict__ ru, const wreal* __restrict__ rud,                   \
    const wreal* __restrict__ rv, const wreal* __restrict__ rvd,                   \
    const wreal* __restrict__ rw, const wreal* __restrict__ rwd,                   \
    const wreal* __restrict__ mut, const wreal* __restrict__ mutd,                 \
    const wreal* __restrict__ c1h, const wreal* __restrict__ c2h,                  \
    const wreal* __restrict__ rdnw, const wreal* __restrict__ fnm,                 \
    const wreal* __restrict__ fnp, const wreal* __restrict__ msft,                 \
    wreal dx, wreal dy, wreal dt,                                                  \
    wreal* __restrict__ fxl, wreal* __restrict__ fxld,                             \
    wreal* __restrict__ fxc, wreal* __restrict__ fxcd,                             \
    wreal* __restrict__ fyl, wreal* __restrict__ fyld,                             \
    wreal* __restrict__ fyc, wreal* __restrict__ fycd,                             \
    wreal* __restrict__ fzl, wreal* __restrict__ fzld,                             \
    wreal* __restrict__ fzc, wreal* __restrict__ fzcd,                             \
    int nz, int ny, int nx, int has_msf, int open_x, int open_y, int vorder

#define LD(a, ad, at) dl((a)[at], (ad)[at])

static __device__ __forceinline__ void put(wreal* v, wreal* d, size_t at, Dl x)
{
    v[at] = x.v;
    if (WRFPLUS_TL) d[at] = x.d;
}

// pd_fluxes: one thread per face index (i 0..nx, j 0..ny, k 0..nz)
extern "C" __global__ void pd_flux_tl(FLUX_TL_ARGS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (i > nx || j > ny || k > nz) return;
    if (j < ny && k < nz) {                                   // x face i
        size_t at = R3(k, j, i, ny, nx + 1);
        HFace h = hface(i, nx, open_x);
        if (!h.live) {
            put(fxl, fxld, at, cst(0));
            put(fxc, fxcd, at, cst(0));
        } else {
            Dl vel = LD(ru, rud, at);
            S6 s;
            for (int o = 0; o < 6; ++o)
                s.s[o] = h.cell[o] >= 0 && hread(h.band, o)
                     ? LD(q, qd, R3(k, j, h.cell[o], ny, nx)) : cst(0);
            Dl fl = hface_low(vel, LD(mut, mutd, (size_t)j * nx + h.r),
                              LD(mut, mutd, (size_t)j * nx + h.l), c1h[k], c2h[k],
                              msft[(size_t)j * nx + h.r], msft[(size_t)j * nx + h.l],
                              has_msf, dx, dt, LD(q0, q0d, R3(k, j, h.l, ny, nx)),
                              LD(q0, q0d, R3(k, j, h.r, ny, nx)),
                              pd_west_wrf(open_x) && i == 1);
            Dl fh = hface_high(s, vel, h.band);
            put(fxl, fxld, at, fl);
            put(fxc, fxcd, at, fh - fl);
        }
    }
    if (i < nx && k < nz) {                                   // y face j
        size_t at = R3(k, j, i, ny + 1, nx);
        HFace h = hface(j, ny, open_y);
        if (!h.live) {
            put(fyl, fyld, at, cst(0));
            put(fyc, fycd, at, cst(0));
        } else {
            Dl vel = LD(rv, rvd, at);
            S6 s;
            for (int o = 0; o < 6; ++o)
                s.s[o] = h.cell[o] >= 0 && hread(h.band, o)
                     ? LD(q, qd, R3(k, h.cell[o], i, ny, nx)) : cst(0);
            Dl fl = hface_low(vel, LD(mut, mutd, (size_t)h.r * nx + i),
                              LD(mut, mutd, (size_t)h.l * nx + i), c1h[k], c2h[k],
                              msft[(size_t)h.r * nx + i], msft[(size_t)h.l * nx + i],
                              has_msf, dy, dt, LD(q0, q0d, R3(k, h.l, i, ny, nx)),
                              LD(q0, q0d, R3(k, h.r, i, ny, nx)), 0);
            Dl fh = hface_high(s, vel, h.band);
            put(fyl, fyld, at, fl);
            put(fyc, fycd, at, fh - fl);
        }
    }
    if (i < nx && j < ny) {                                   // eta face k
        size_t at = R3(k, j, i, ny, nx);
        if (k == 0 || k == nz) {
            put(fzl, fzld, at, cst(0));
            put(fzc, fzcd, at, cst(0));
        } else {
            Dl vel = LD(rw, rwd, at);
            S6 s;
            for (int o = 0; o < 6; ++o)
                s.s[o] = zread(k, nz, vorder, o) ? LD(q, qd, R3(k - 3 + o, j, i, ny, nx)) : cst(0);
            Dl fl = zface_low(vel, LD(mut, mutd, (size_t)j * nx + i), c1h[k], c2h[k], rdnw[k],
                              rdnw[k - 1], dt, LD(q0, q0d, R3(k - 1, j, i, ny, nx)),
                              LD(q0, q0d, R3(k, j, i, ny, nx)));
            Dl fh = zface_high(s, vel, k, nz, fnm[k], fnp[k], vorder);
            put(fzl, fzld, at, fl);
            put(fzc, fzcd, at, fh - fl);
        }
    }
}

// The cell's ph_low and flux_out (pd_scale's statements) on value/tangent
// pairs; ``fx`` etc. are the six faces' fluxes.
struct Budget { Dl ph; Dl fo; };

static __device__ __forceinline__ Budget budget(Dl q0, Dl mup0, wreal mub, wreal c1, wreal c2,
                                                wreal rdk, wreal m, int has_msf,
                                                wreal dx_inv, wreal dy_inv, wreal dt,
                                                Dl xl0, Dl xl1, Dl yl0, Dl yl1, Dl zl0, Dl zl1,
                                                Dl xc0, Dl xc1, Dl yc0, Dl yc1, Dl zc0, Dl zc1)
{
    Budget b;
    Dl chm0 = (c1 * mub + c2) + c1 * mup0;
    if (has_msf) {
        wreal m2 = m * m;
        b.ph = chm0 * q0 - dt * (m2 * (dx_inv * (xl1 - xl0) + dy_inv * (yl1 - yl0))
                                 + (m * rdk) * (zl1 - zl0));
        b.fo = dt * (m2 * (dx_inv * (dmax0(xc1) - dmin0(xc0))
                           + dy_inv * (dmax0(yc1) - dmin0(yc0)))
                     + (m * rdk) * (dmin0(zc1) - dmax0(zc0)));
    } else {
        Dl dlo = dx_inv * (xl1 - xl0) + dy_inv * (yl1 - yl0) + rdk * (zl1 - zl0);
        b.ph = chm0 * q0 - dt * dlo;
        b.fo = dt * (dx_inv * (dmax0(xc1) - dmin0(xc0)) + dy_inv * (dmax0(yc1) - dmin0(yc0))
                     + rdk * (dmin0(zc1) - dmax0(zc0)));
    }
    return b;
}

// limited: the cell renormalises (fo > ph); then max(0, ph/(fo+eps))
static __device__ __forceinline__ Dl scale_of(Budget b, int* limited)
{
    *limited = 0;
    if (!(b.fo.v > b.ph.v)) return cst(1);
    Dl r = ddiv(b.ph, b.fo + PD_EPS);
    if ((wreal)0 < r.v) {
        *limited = 1;
        return r;
    }
    return cst(wfmax((wreal)0, r.v));
}

static __device__ __forceinline__ int excluded(int j, int i, int ny, int nx, int open_x,
                                               int open_y)
{
    return (open_x && (i == 0 || i == nx - 1)) || (open_y && (j == 0 || j == ny - 1));
}

#define SCALE_ARGS                                                                 \
    const wreal* __restrict__ q0, const wreal* __restrict__ q0d,                   \
    const wreal* __restrict__ mup0, const wreal* __restrict__ mup0d,               \
    const wreal* __restrict__ mub, const wreal* __restrict__ c1h,                  \
    const wreal* __restrict__ c2h, const wreal* __restrict__ rdnw,                 \
    const wreal* __restrict__ msft,                                                \
    const wreal* __restrict__ fxl, const wreal* __restrict__ fxld,                 \
    const wreal* __restrict__ fxc, const wreal* __restrict__ fxcd,                 \
    const wreal* __restrict__ fyl, const wreal* __restrict__ fyld,                 \
    const wreal* __restrict__ fyc, const wreal* __restrict__ fycd,                 \
    const wreal* __restrict__ fzl, const wreal* __restrict__ fzld,                 \
    const wreal* __restrict__ fzc, const wreal* __restrict__ fzcd,                 \
    wreal dx_inv, wreal dy_inv, wreal dt,                                          \
    wreal* __restrict__ sc, wreal* __restrict__ scd,                               \
    int nz, int ny, int nx, int has_msf, int open_x, int open_y

#define XF(a, ad, k, j, f) LD(a, ad, R3((k), (j), (f), ny, nx + 1))
#define YF(a, ad, k, g, i) LD(a, ad, R3((k), (g), (i), ny + 1, nx))
#define ZF(a, ad, kf, j, i) LD(a, ad, R3((kf), (j), (i), ny, nx))

static __device__ __forceinline__ Budget cell_budget(
    int k, int j, int i, const wreal* q0, const wreal* q0d, const wreal* mup0,
    const wreal* mup0d, const wreal* mub, const wreal* c1h, const wreal* c2h,
    const wreal* rdnw, const wreal* msft, const wreal* fxl, const wreal* fxld,
    const wreal* fxc, const wreal* fxcd, const wreal* fyl, const wreal* fyld,
    const wreal* fyc, const wreal* fycd, const wreal* fzl, const wreal* fzld,
    const wreal* fzc, const wreal* fzcd, wreal dx_inv, wreal dy_inv, wreal dt,
    int ny, int nx, int has_msf)
{
    size_t col = (size_t)j * nx + i;
    return budget(LD(q0, q0d, R3(k, j, i, ny, nx)), LD(mup0, mup0d, col), mub[col], c1h[k],
                  c2h[k], rdnw[k], has_msf ? msft[col] : (wreal)1, has_msf, dx_inv, dy_inv, dt,
                  XF(fxl, fxld, k, j, i), XF(fxl, fxld, k, j, i + 1),
                  YF(fyl, fyld, k, j, i), YF(fyl, fyld, k, j + 1, i),
                  ZF(fzl, fzld, k, j, i), ZF(fzl, fzld, k + 1, j, i),
                  XF(fxc, fxcd, k, j, i), XF(fxc, fxcd, k, j, i + 1),
                  YF(fyc, fycd, k, j, i), YF(fyc, fycd, k, j + 1, i),
                  ZF(fzc, fzcd, k, j, i), ZF(fzc, fzcd, k + 1, j, i));
}

#define CELL_BUDGET(k, j, i)                                                       \
    cell_budget((k), (j), (i), q0, q0d, mup0, mup0d, mub, c1h, c2h, rdnw, msft,    \
                fxl, fxld, fxc, fxcd, fyl, fyld, fyc, fycd, fzl, fzld, fzc, fzcd,  \
                dx_inv, dy_inv, dt, ny, nx, has_msf)

extern "C" __global__ void pd_scale_tl(SCALE_ARGS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    size_t at = R3(k, j, i, ny, nx);
    Dl s = cst(1);
    if (!excluded(j, i, ny, nx, open_x, open_y)) {
        int limited;
        s = scale_of(CELL_BUDGET(k, j, i), &limited);
    }
    put(sc, scd, at, s);
}

// The donor's factor for a face with correction fc, between cells ``lo``
// (fc > 0 drains it) and ``hi`` (fc < 0); 1 for a zero correction.
static __device__ __forceinline__ Dl donor(const wreal* sc, const wreal* scd, wreal fc,
                                           size_t lo, size_t hi)
{
    if (fc > (wreal)0) return LD(sc, scd, lo);
    if (fc < (wreal)0) return LD(sc, scd, hi);
    return cst(1);
}

#define APPLY_TL_ARGS                                                              \
    const wreal* __restrict__ fxl, const wreal* __restrict__ fxld,                 \
    const wreal* __restrict__ fxc, const wreal* __restrict__ fxcd,                 \
    const wreal* __restrict__ fyl, const wreal* __restrict__ fyld,                 \
    const wreal* __restrict__ fyc, const wreal* __restrict__ fycd,                 \
    const wreal* __restrict__ fzl, const wreal* __restrict__ fzld,                 \
    const wreal* __restrict__ fzc, const wreal* __restrict__ fzcd,                 \
    const wreal* __restrict__ sc, const wreal* __restrict__ scd,                   \
    const wreal* __restrict__ rdnw, const wreal* __restrict__ msft,                \
    wreal dx_inv, wreal dy_inv,                                                    \
    wreal* __restrict__ tend, wreal* __restrict__ tendd,                           \
    int nz, int ny, int nx, int has_msf, int open_x, int open_y

// pd_renorm_apply (zero_first): z, then x, then y, as the native limiter
extern "C" __global__ void pd_apply_tl(APPLY_TL_ARGS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
#define SC(kk, jj, ii) R3((kk), (jj), (ii), ny, nx)
    size_t me = SC(k, j, i);
    Dl fxc_l = XF(fxc, fxcd, k, j, i), fxc_r = XF(fxc, fxcd, k, j, i + 1);
    Dl sx_l = donor(sc, scd, fxc_l.v, SC(k, j, pwrap(i - 1, nx)), me);
    Dl sx_r = donor(sc, scd, fxc_r.v, me, SC(k, j, pwrap(i + 1, nx)));
    Dl fyc_l = YF(fyc, fycd, k, j, i), fyc_r = YF(fyc, fycd, k, j + 1, i);
    Dl sy_l = donor(sc, scd, fyc_l.v, SC(k, pwrap(j - 1, ny), i), me);
    Dl sy_r = donor(sc, scd, fyc_r.v, me, SC(k, pwrap(j + 1, ny), i));
    // eta: positive (downward) drains the upper cell, negative the lower
    Dl fzc_b = ZF(fzc, fzcd, k, j, i), fzc_t = ZF(fzc, fzcd, k + 1, j, i);
    Dl sz_b = donor(sc, scd, fzc_b.v, me, k > 0 ? SC(k - 1, j, i) : me);
    Dl sz_t = donor(sc, scd, -fzc_t.v, me, k + 1 < nz ? SC(k + 1, j, i) : me);
#undef SC
    Dl result = cst(0);
    Dl zb = sz_t * fzc_t - sz_b * fzc_b + ZF(fzl, fzld, k + 1, j, i) - ZF(fzl, fzld, k, j, i);
    result = result - rdnw[k] * zb;
    wreal m = has_msf ? msft[(size_t)j * nx + i] : (wreal)1;
    if (!open_x || (i >= 1 && i <= nx - 2)) {
        Dl xb = sx_r * fxc_r - sx_l * fxc_l + XF(fxl, fxld, k, j, i + 1) - XF(fxl, fxld, k, j, i);
        result = result - m * (dx_inv * xb);
    }
    if (!open_y || (j >= 1 && j <= ny - 2)) {
        Dl yb = sy_r * fyc_r - sy_l * fyc_l + YF(fyl, fyld, k, j + 1, i) - YF(fyl, fyld, k, j, i);
        result = result - m * (dy_inv * yb);
    }
    put(tend, tendd, me, result);
}

// ----------------------------------------------------------------- AD

// a short list of face indices, by value
struct FList { int f[14]; int n; };

static __device__ __forceinline__ int applies(int c, int n, int open)
{
    return !open || (c >= 1 && c <= n - 2);
}

// pd_apply's transpose per face: g the adjoint of the face's total flux
// fl + s*fc; afl += g, afc += s*g, gs = fc*g for the donor's factor.
extern "C" __global__ void pd_apply_ad(
    const wreal* __restrict__ at, const wreal* __restrict__ sc,
    const wreal* __restrict__ fxc, const wreal* __restrict__ fyc,
    const wreal* __restrict__ fzc, const wreal* __restrict__ rdnw,
    const wreal* __restrict__ msft, wreal dx_inv, wreal dy_inv,
    wreal* __restrict__ afxl, wreal* __restrict__ afxc, wreal* __restrict__ gsx,
    wreal* __restrict__ afyl, wreal* __restrict__ afyc, wreal* __restrict__ gsy,
    wreal* __restrict__ afzl, wreal* __restrict__ afzc, wreal* __restrict__ gsz,
    int nz, int ny, int nx, int has_msf, int open_x, int open_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (i > nx || j > ny || k > nz) return;
#define M(jj, ii) (has_msf ? msft[(size_t)(jj) * nx + (ii)] : (wreal)1)
    if (j < ny && k < nz) {                                   // x face i
        size_t f = R3(k, j, i, ny, nx + 1);
        wreal g = 0;
        if (i >= 1 && applies(i - 1, nx, open_x))
            g += -(M(j, i - 1) * dx_inv) * at[R3(k, j, i - 1, ny, nx)];
        if (i <= nx - 1 && applies(i, nx, open_x))
            g += (M(j, i) * dx_inv) * at[R3(k, j, i, ny, nx)];
        wreal c = fxc[f];
        wreal s = c > (wreal)0 ? sc[R3(k, j, pwrap(i - 1, nx), ny, nx)]
                : (c < (wreal)0 ? sc[R3(k, j, pwrap(i, nx), ny, nx)] : (wreal)1);
        afxl[f] = g;
        afxc[f] = s * g;
        gsx[f] = c * g;
    }
    if (i < nx && k < nz) {                                   // y face j
        size_t f = R3(k, j, i, ny + 1, nx);
        wreal g = 0;
        if (j >= 1 && applies(j - 1, ny, open_y))
            g += -(M(j - 1, i) * dy_inv) * at[R3(k, j - 1, i, ny, nx)];
        if (j <= ny - 1 && applies(j, ny, open_y))
            g += (M(j, i) * dy_inv) * at[R3(k, j, i, ny, nx)];
        wreal c = fyc[f];
        wreal s = c > (wreal)0 ? sc[R3(k, pwrap(j - 1, ny), i, ny, nx)]
                : (c < (wreal)0 ? sc[R3(k, pwrap(j, ny), i, ny, nx)] : (wreal)1);
        afyl[f] = g;
        afyc[f] = s * g;
        gsy[f] = c * g;
    }
    if (i < nx && j < ny) {                                   // eta face k
        size_t f = R3(k, j, i, ny, nx);
        wreal g = 0;
        if (k >= 1) g += -rdnw[k - 1] * at[R3(k - 1, j, i, ny, nx)];
        if (k <= nz - 1) g += rdnw[k] * at[R3(k, j, i, ny, nx)];
        wreal c = fzc[f];
        wreal s = 1;
        if (c > (wreal)0 && k <= nz - 1) s = sc[R3(k, j, i, ny, nx)];
        if (c < (wreal)0 && k >= 1) s = sc[R3(k - 1, j, i, ny, nx)];
        afzl[f] = g;
        afzc[f] = s * g;
        gsz[f] = c * g;
    }
#undef M
}

// The faces of cell c along an axis of n cells whose correction the cell
// donates to: its own two faces, and on a periodic axis the seam's other
// copy (face n aliases face 0).  Returns the count, faces in ``out``.
static __device__ __forceinline__ FList own_faces(int c, int n, int open)
{
    FList r;
    r.n = 0;
    r.f[r.n++] = c;
    r.f[r.n++] = c + 1;
    if (!open && n > 1) {
        if (c == 0) r.f[r.n++] = n;
        if (c == n - 1) r.f[r.n++] = 0;
    }
    return r;
}

// the donor cell of a horizontal face (-1 for none)
static __device__ __forceinline__ int hdonor(int f, int n, int open, wreal fc)
{
    if (fc > (wreal)0) return open ? f - 1 : pwrap(f - 1, n);
    if (fc < (wreal)0) return open ? f : pwrap(f, n);
    return -1;
}

// the factor's adjoint, then ph_low's and flux_out's (aph, afo), and
// q0's and the time-t mass's shares through ph_low (aq0, amu by level)
extern "C" __global__ void pd_scale_ad(
    const wreal* __restrict__ gsx, const wreal* __restrict__ gsy,
    const wreal* __restrict__ gsz,
    const wreal* __restrict__ q0, const wreal* __restrict__ mup0,
    const wreal* __restrict__ mub, const wreal* __restrict__ c1h,
    const wreal* __restrict__ c2h, const wreal* __restrict__ rdnw,
    const wreal* __restrict__ msft,
    const wreal* __restrict__ fxl, const wreal* __restrict__ fxc,
    const wreal* __restrict__ fyl, const wreal* __restrict__ fyc,
    const wreal* __restrict__ fzl, const wreal* __restrict__ fzc,
    wreal dx_inv, wreal dy_inv, wreal dt,
    wreal* __restrict__ aph, wreal* __restrict__ afo,
    wreal* __restrict__ aq0, wreal* __restrict__ amu,
    int nz, int ny, int nx, int has_msf, int open_x, int open_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    size_t me = R3(k, j, i, ny, nx);
    wreal a_ph = 0, a_fo = 0;
    if (!excluded(j, i, ny, nx, open_x, open_y)) {
        Budget b = budget(dl(q0[me], 0), dl(mup0[(size_t)j * nx + i], 0),
                          mub[(size_t)j * nx + i], c1h[k], c2h[k], rdnw[k],
                          has_msf ? msft[(size_t)j * nx + i] : (wreal)1, has_msf,
                          dx_inv, dy_inv, dt,
                          cst(fxl[R3(k, j, i, ny, nx + 1)]), cst(fxl[R3(k, j, i + 1, ny, nx + 1)]),
                          cst(fyl[R3(k, j, i, ny + 1, nx)]), cst(fyl[R3(k, j + 1, i, ny + 1, nx)]),
                          cst(fzl[R3(k, j, i, ny, nx)]), cst(fzl[R3(k + 1, j, i, ny, nx)]),
                          cst(fxc[R3(k, j, i, ny, nx + 1)]), cst(fxc[R3(k, j, i + 1, ny, nx + 1)]),
                          cst(fyc[R3(k, j, i, ny + 1, nx)]), cst(fyc[R3(k, j + 1, i, ny + 1, nx)]),
                          cst(fzc[R3(k, j, i, ny, nx)]), cst(fzc[R3(k + 1, j, i, ny, nx)]));
        int limited;
        Dl r = scale_of(b, &limited);
        if (limited) {
            wreal as = 0;
            FList fs = own_faces(i, nx, open_x);
            for (int t = 0; t < fs.n; ++t) {
                size_t f = R3(k, j, fs.f[t], ny, nx + 1);
                if (hdonor(fs.f[t], nx, open_x, fxc[f]) == i) as += gsx[f];
            }
            fs = own_faces(j, ny, open_y);
            for (int t = 0; t < fs.n; ++t) {
                size_t f = R3(k, fs.f[t], i, ny + 1, nx);
                if (hdonor(fs.f[t], ny, open_y, fyc[f]) == j) as += gsy[f];
            }
            if (fzc[R3(k, j, i, ny, nx)] > (wreal)0) as += gsz[R3(k, j, i, ny, nx)];
            if (fzc[R3(k + 1, j, i, ny, nx)] < (wreal)0) as += gsz[R3(k + 1, j, i, ny, nx)];
            wreal den = b.fo.v + PD_EPS;
            a_ph = as / den;
            a_fo = -(as * r.v) / den;
        }
    }
    aph[me] = a_ph;
    afo[me] = a_fo;
    // ph_low = ((c1*mub + c2) + c1*mup0)*q0 - ...
    wreal chm0 = (c1h[k] * mub[(size_t)j * nx + i] + c2h[k]) + c1h[k] * mup0[(size_t)j * nx + i];
    aq0[me] = chm0 * a_ph;
    amu[me] = c1h[k] * q0[me] * a_ph;
}

// Per face: the limiter's share of the face's adjoints (ph_low and
// flux_out of the two cells that read it), then the face flux's own
// partial in its mass flux; leaves afh (in afc) and afl (in afl) for
// pd_flux_ad_cell.
extern "C" __global__ void pd_flux_ad_face(
    const wreal* __restrict__ q, const wreal* __restrict__ q0,
    const wreal* __restrict__ ru, const wreal* __restrict__ rv,
    const wreal* __restrict__ rw, const wreal* __restrict__ mut,
    const wreal* __restrict__ c1h, const wreal* __restrict__ c2h,
    const wreal* __restrict__ rdnw, const wreal* __restrict__ fnm,
    const wreal* __restrict__ fnp, const wreal* __restrict__ msft,
    const wreal* __restrict__ fxc, const wreal* __restrict__ fyc,
    const wreal* __restrict__ fzc,
    const wreal* __restrict__ aph, const wreal* __restrict__ afo,
    wreal dx, wreal dy, wreal dt, wreal dx_inv, wreal dy_inv,
    wreal* __restrict__ afxl, wreal* __restrict__ afxc,
    wreal* __restrict__ afyl, wreal* __restrict__ afyc,
    wreal* __restrict__ afzl, wreal* __restrict__ afzc,
    wreal* __restrict__ aru, wreal* __restrict__ arv, wreal* __restrict__ arw,
    int nz, int ny, int nx, int has_msf, int open_x, int open_y, int vorder)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (i > nx || j > ny || k > nz) return;
#define M(jj, ii) (has_msf ? msft[(size_t)(jj) * nx + (ii)] : (wreal)1)
    if (j < ny && k < nz) {                                   // x face i
        size_t f = R3(k, j, i, ny, nx + 1);
        HFace h = hface(i, nx, open_x);
        wreal al = afxl[f], ac = afxc[f], c = fxc[f];
        if (i >= 1) {                       // cell i-1 reads it as its right face
            size_t cl = R3(k, j, i - 1, ny, nx);
            wreal w = dt * (M(j, i - 1) * M(j, i - 1)) * dx_inv;
            al += -w * aph[cl];
            if (c > (wreal)0) ac += w * afo[cl];
        }
        if (i <= nx - 1) {                  // cell i as its left face
            size_t cr = R3(k, j, i, ny, nx);
            wreal w = dt * (M(j, i) * M(j, i)) * dx_inv;
            al += w * aph[cr];
            if (c < (wreal)0) ac += -w * afo[cr];
        }
        wreal afh = 0, afl = 0;
        if (h.live) {
            afh = ac;
            afl = al - ac;
            S6 s;
            for (int o = 0; o < 6; ++o)
                s.s[o] = cst(h.cell[o] >= 0 && hread(h.band, o) ? q[R3(k, j, h.cell[o], ny, nx)] : 0);
            Dl vel = dl(ru[f], 1);
            Dl fl = hface_low(vel, cst(mut[(size_t)j * nx + h.r]), cst(mut[(size_t)j * nx + h.l]),
                              c1h[k], c2h[k], msft[(size_t)j * nx + h.r],
                              msft[(size_t)j * nx + h.l], has_msf, dx, dt,
                              cst(q0[R3(k, j, h.l, ny, nx)]), cst(q0[R3(k, j, h.r, ny, nx)]),
                              pd_west_wrf(open_x) && i == 1);
            Dl fh = hface_high(s, vel, h.band);
            aru[f] += afh * fh.d + afl * fl.d;
        }
        afxc[f] = afh;
        afxl[f] = afl;
    }
    if (i < nx && k < nz) {                                   // y face j
        size_t f = R3(k, j, i, ny + 1, nx);
        HFace h = hface(j, ny, open_y);
        wreal al = afyl[f], ac = afyc[f], c = fyc[f];
        if (j >= 1) {
            size_t cl = R3(k, j - 1, i, ny, nx);
            wreal w = dt * (M(j - 1, i) * M(j - 1, i)) * dy_inv;
            al += -w * aph[cl];
            if (c > (wreal)0) ac += w * afo[cl];
        }
        if (j <= ny - 1) {
            size_t cr = R3(k, j, i, ny, nx);
            wreal w = dt * (M(j, i) * M(j, i)) * dy_inv;
            al += w * aph[cr];
            if (c < (wreal)0) ac += -w * afo[cr];
        }
        wreal afh = 0, afl = 0;
        if (h.live) {
            afh = ac;
            afl = al - ac;
            S6 s;
            for (int o = 0; o < 6; ++o)
                s.s[o] = cst(h.cell[o] >= 0 && hread(h.band, o) ? q[R3(k, h.cell[o], i, ny, nx)] : 0);
            Dl vel = dl(rv[f], 1);
            Dl fl = hface_low(vel, cst(mut[(size_t)h.r * nx + i]), cst(mut[(size_t)h.l * nx + i]),
                              c1h[k], c2h[k], msft[(size_t)h.r * nx + i],
                              msft[(size_t)h.l * nx + i], has_msf, dy, dt,
                              cst(q0[R3(k, h.l, i, ny, nx)]), cst(q0[R3(k, h.r, i, ny, nx)]), 0);
            Dl fh = hface_high(s, vel, h.band);
            arv[f] += afh * fh.d + afl * fl.d;
        }
        afyc[f] = afh;
        afyl[f] = afl;
    }
    if (i < nx && j < ny) {                                   // eta face k
        size_t f = R3(k, j, i, ny, nx);
        wreal al = afzl[f], ac = afzc[f], c = fzc[f];
        wreal m = M(j, i);
        if (k >= 1) {                       // cell k-1 reads it as its top
            size_t cl = R3(k - 1, j, i, ny, nx);
            wreal w = dt * (m * rdnw[k - 1]);
            al += -w * aph[cl];
            if (c < (wreal)0) ac += w * afo[cl];
        }
        if (k <= nz - 1) {                  // cell k as its bottom
            size_t cu = R3(k, j, i, ny, nx);
            wreal w = dt * (m * rdnw[k]);
            al += w * aph[cu];
            if (c > (wreal)0) ac += -w * afo[cu];
        }
        wreal afh = 0, afl = 0;
        if (k > 0 && k < nz) {
            afh = ac;
            afl = al - ac;
            S6 s;
            for (int o = 0; o < 6; ++o)
                s.s[o] = cst(zread(k, nz, vorder, o) ? q[R3(k - 3 + o, j, i, ny, nx)] : 0);
            Dl vel = dl(rw[f], 1);
            Dl fl = zface_low(vel, cst(mut[(size_t)j * nx + i]), c1h[k], c2h[k], rdnw[k],
                              rdnw[k - 1], dt, cst(q0[R3(k - 1, j, i, ny, nx)]),
                              cst(q0[R3(k, j, i, ny, nx)]));
            Dl fh = zface_high(s, vel, k, nz, fnm[k], fnp[k], vorder);
            arw[f] += afh * fh.d + afl * fl.d;
        }
        afzc[f] = afh;
        afzl[f] = afl;
    }
#undef M
}

// The faces along an axis of n cells whose high-order stencil can read
// cell c (faces c-2 .. c+3, wrapped and with the seam copy on a periodic
// axis, deduplicated); returns the count.
static __device__ __forceinline__ FList stencil_faces(int c, int n, int open)
{
    FList r;
    r.n = 0;
    for (int o = -2; o <= 3; ++o) {
        int f = c + o;
        if (open) {
            if (f >= 1 && f <= n - 1) r.f[r.n++] = f;
            continue;
        }
        f = pwrap(f, n);
        for (int rep = 0; rep < 2; ++rep) {
            int g = rep == 0 ? f : (f == 0 ? n : -1);
            if (g < 0) continue;
            int dup = 0;
            for (int t = 0; t < r.n; ++t) dup |= r.f[t] == g;
            if (!dup) r.f[r.n++] = g;
        }
    }
    return r;
}

// Per cell: q's adjoint from every high-order face that reads it, q0's
// from the upwind fluxes (added to pd_scale_ad's aq0) and the column
// mass's partial by level (amut, summed over k by the caller).
extern "C" __global__ void pd_flux_ad_cell(
    const wreal* __restrict__ q, const wreal* __restrict__ q0,
    const wreal* __restrict__ ru, const wreal* __restrict__ rv,
    const wreal* __restrict__ rw, const wreal* __restrict__ mut,
    const wreal* __restrict__ c1h, const wreal* __restrict__ c2h,
    const wreal* __restrict__ rdnw, const wreal* __restrict__ fnm,
    const wreal* __restrict__ fnp, const wreal* __restrict__ msft,
    const wreal* __restrict__ afxl, const wreal* __restrict__ afxc,
    const wreal* __restrict__ afyl, const wreal* __restrict__ afyc,
    const wreal* __restrict__ afzl, const wreal* __restrict__ afzc,
    wreal dx, wreal dy, wreal dt,
    wreal* __restrict__ aq, wreal* __restrict__ aq0, wreal* __restrict__ amut,
    int nz, int ny, int nx, int has_msf, int open_x, int open_y, int vorder)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    size_t me = R3(k, j, i, ny, nx);
    wreal a_q = 0, a_q0 = 0, a_mu = 0;
    // x faces
    FList fs = stencil_faces(i, nx, open_x);
    for (int t = 0; t < fs.n; ++t) {
        int f = fs.f[t];
        HFace h = hface(f, nx, open_x);
        if (!h.live) continue;
        size_t at = R3(k, j, f, ny, nx + 1);
        Dl vel = cst(ru[at]);
        S6 s;
        int hit = 0;
        for (int o = 0; o < 6; ++o) {
            int rd = h.cell[o] >= 0 && hread(h.band, o);
            int me_o = rd && h.cell[o] == i;
            hit |= me_o;
            s.s[o] = dl(rd ? q[R3(k, j, h.cell[o], ny, nx)] : 0, me_o ? 1 : 0);
        }
        if (hit) a_q += afxc[at] * hface_high(s, vel, h.band).d;
        int west = pd_west_wrf(open_x) && f == 1;
        wreal mr = mut[(size_t)j * nx + h.r], ml = mut[(size_t)j * nx + h.l];
        wreal sr = msft[(size_t)j * nx + h.r], sl = msft[(size_t)j * nx + h.l];
        wreal q0l = q0[R3(k, j, h.l, ny, nx)], q0r = q0[R3(k, j, h.r, ny, nx)];
        if (h.l == i || h.r == i) {
            wreal afl = afxl[at];
            Dl fl = hface_low(vel, cst(mr), cst(ml), c1h[k], c2h[k], sr, sl, has_msf, dx, dt,
                              dl(q0l, h.l == i ? 1 : 0), dl(q0r, h.r == i ? 1 : 0), west);
            a_q0 += afl * fl.d;
            fl = hface_low(vel, dl(mr, h.r == i ? 1 : 0), dl(ml, h.l == i ? 1 : 0), c1h[k],
                           c2h[k], sr, sl, has_msf, dx, dt, cst(q0l), cst(q0r), west);
            a_mu += afl * fl.d;
        }
    }
    // y faces
    fs = stencil_faces(j, ny, open_y);
    for (int t = 0; t < fs.n; ++t) {
        int g = fs.f[t];
        HFace h = hface(g, ny, open_y);
        if (!h.live) continue;
        size_t at = R3(k, g, i, ny + 1, nx);
        Dl vel = cst(rv[at]);
        S6 s;
        int hit = 0;
        for (int o = 0; o < 6; ++o) {
            int rd = h.cell[o] >= 0 && hread(h.band, o);
            int me_o = rd && h.cell[o] == j;
            hit |= me_o;
            s.s[o] = dl(rd ? q[R3(k, h.cell[o], i, ny, nx)] : 0, me_o ? 1 : 0);
        }
        if (hit) a_q += afyc[at] * hface_high(s, vel, h.band).d;
        wreal mr = mut[(size_t)h.r * nx + i], ml = mut[(size_t)h.l * nx + i];
        wreal sr = msft[(size_t)h.r * nx + i], sl = msft[(size_t)h.l * nx + i];
        wreal q0l = q0[R3(k, h.l, i, ny, nx)], q0r = q0[R3(k, h.r, i, ny, nx)];
        if (h.l == j || h.r == j) {
            wreal afl = afyl[at];
            Dl fl = hface_low(vel, cst(mr), cst(ml), c1h[k], c2h[k], sr, sl, has_msf, dy, dt,
                              dl(q0l, h.l == j ? 1 : 0), dl(q0r, h.r == j ? 1 : 0), 0);
            a_q0 += afl * fl.d;
            fl = hface_low(vel, dl(mr, h.r == j ? 1 : 0), dl(ml, h.l == j ? 1 : 0), c1h[k],
                           c2h[k], sr, sl, has_msf, dy, dt, cst(q0l), cst(q0r), 0);
            a_mu += afl * fl.d;
        }
    }
    // eta faces k-2 .. k+3 (live: 1 .. nz-1)
    for (int kf = max(k - 2, 1); kf <= min(k + 3, nz - 1); ++kf) {
        size_t at = R3(kf, j, i, ny, nx);
        Dl vel = cst(rw[at]);
        S6 s;
        int hit = 0;
        for (int o = 0; o < 6; ++o) {
            int rd = zread(kf, nz, vorder, o);
            int me_o = rd && kf - 3 + o == k;
            hit |= me_o;
            s.s[o] = dl(rd ? q[R3(kf - 3 + o, j, i, ny, nx)] : 0, me_o ? 1 : 0);
        }
        if (hit) a_q += afzc[at] * zface_high(s, vel, kf, nz, fnm[kf], fnp[kf], vorder).d;
        if (kf == k || kf == k + 1) {
            wreal afl = afzl[at];
            wreal q0l = q0[R3(kf - 1, j, i, ny, nx)], q0u = q0[R3(kf, j, i, ny, nx)];
            Dl fl = zface_low(vel, cst(mut[(size_t)j * nx + i]), c1h[kf], c2h[kf], rdnw[kf],
                              rdnw[kf - 1], dt, dl(q0l, kf - 1 == k ? 1 : 0),
                              dl(q0u, kf == k ? 1 : 0));
            a_q0 += afl * fl.d;
        }
        if (kf == k) {                      // the face at level k reads the column mass
            Dl fl = zface_low(vel, dl(mut[(size_t)j * nx + i], 1), c1h[kf], c2h[kf], rdnw[kf],
                              rdnw[kf - 1], dt, cst(q0[R3(kf - 1, j, i, ny, nx)]),
                              cst(q0[R3(kf, j, i, ny, nx)]));
            a_mu += afzl[at] * fl.d;
        }
    }
    aq[me] += a_q;
    aq0[me] += a_q0;
    amut[me] = a_mu;
}
