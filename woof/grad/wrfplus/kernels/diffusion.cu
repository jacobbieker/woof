// Tangent-linear and adjoint of woof's strict diff_opt 1 / km_opt 4
// coefficients (P3a item 5):
//
//   deformation  woof wrf_smag_deform (woof/core/kernels/smag2d.cu, the
//                GPUWM_WRF_EXACT_C_DIFFUSION path every diffusion module
//                compiles; wrf_smag_deform_hat gives the same words): WRF
//                compute_diff_metrics (z, rdzw, zx, zy from phb + ph') and
//                cal_deform_and_div's defor11, defor22 and defor12, 3-D
//                base geopotential;
//   smag2d_km    woof wrf_diff_opt1_km4 (diff_opt1.cu) then wrf_smag_km_bc:
//                WRF smag2d_km at diff_opt 1 (no slope limiter), with the
//                cold-zero outer row of a bounded axis.
//
// The oracle is WRFPLUS 4.7.1 g_/a_compute_diff_metrics +
// g_/a_cal_deform_and_div and G_/A_SMAG2D_KM; see
// docs/dev/P3A-WRF-DRYCORE-ADJOINT.md.
//
// wreal is binary64 unless the unit is built with -DWRFPLUS_REAL=float.
// The forward statements are the strict primal's on a value/tangent pair
// (Dl), in the primal's order, so the float instantiation's base state
// gives the primal's words ('+', '-', '*' are the primal's __f*_rn only
// with --fmad=false, which the wrapper passes; division is __fdiv_rn).
//
// One forward serves the TL and the AD.  The TL reads the tangents of the
// active inputs; the AD seeds one input word with tangent 1 (all others
// 0), so a forward's tangent is that output's partial derivative with
// respect to the word.  The AD thread of an input word gathers
// sum_out a_out * d out / d word over the outputs whose stencil can read
// the word (a window of distinct outputs; a word read twice by one output,
// e.g. on a one-cell periodic axis, is summed by the forward itself): one
// writer per word, no atomics, deterministic.  The outputs overwrite, so
// their adjoints are consumed (zeroed) by a second launch.
//
// Layout (k, j, i): mass (nz, ny, nx), u (nz, ny, nx+1), v (nz, ny+1, nx),
// w levels (nz+1, ny, nx), columns (ny, nx).

#ifndef WRFPLUS_REAL
#define WRFPLUS_REAL double
#endif

// WRFPLUS_TL 0 builds the base-only forward (model.step_base): every
// perturbation store of the TL kernels is skipped; the base words are
// unchanged.
#ifndef WRFPLUS_TL
#define WRFPLUS_TL 1
#endif
typedef WRFPLUS_REAL wreal;

#define R3(k, j, i, nyy, nxx) \
    ((size_t)(k) * (size_t)(nyy) * (size_t)(nxx) + (size_t)(j) * (size_t)(nxx) + (size_t)(i))

#define DF_G ((wreal)9.81)

static __device__ __forceinline__ float wdiv(float a, float b) { return __fdiv_rn(a, b); }
static __device__ __forceinline__ double wdiv(double a, double b) { return a / b; }
static __device__ __forceinline__ float wsqrt(float a) { return sqrtf(a); }
static __device__ __forceinline__ double wsqrt(double a) { return sqrt(a); }

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
static __device__ __forceinline__ Dl operator*(Dl a, wreal b) { return dl(a.v * b, a.d * b); }
static __device__ __forceinline__ Dl operator+(wreal a, Dl b) { return dl(a + b.v, b.d); }
static __device__ __forceinline__ Dl operator/(Dl a, wreal b)
{
    return dl(wdiv(a.v, b), wdiv(a.d, b));
}
// a / b with b active: the value is the primal's division.
static __device__ __forceinline__ Dl operator/(wreal a, Dl b)
{
    wreal v = wdiv(a, b.v);
    return dl(v, wdiv(-(v * b.d), b.v));
}

static __device__ __forceinline__ int df_per(int i, int n) { return ((i % n) + n) % n; }

// The candidate output coordinate c + off on an axis of n points: the
// periodic image, or -1 when it lies off a bounded axis or repeats the
// image of an earlier offset (lo..off-1), so each output is visited once.
static __device__ int df_cand(int c, int off, int lo, int n, int bounded)
{
    int x = c + off;
    if (bounded) return (x < 0 || x >= n) ? -1 : x;
    x = df_per(x, n);
    for (int o = lo; o < off; ++o)
        if (df_per(c + o, n) == x) return -1;
    return x;
}

// ================================================ deformation (diff metrics)

// Input ids of the seeded (AD) forward.
#define DF_U 1
#define DF_V 2
#define DF_PH 3

struct DfSrc {
    const wreal* u;
    const wreal* ud;
    const wreal* v;
    const wreal* vd;
    const wreal* php;
    const wreal* phpd;
    const wreal* phb;
    const wreal* msft;
    const wreal* msfu;
    const wreal* msfv;
    const wreal* fnm;
    const wreal* fnp;
    const wreal* dn;
    const wreal* dnw;
    wreal rdx, rdy, cf1, cf2, cf3;
    int nz, ny, nx, bx, by;
    int seam_x, seam_y;  // WRF's cold seam slope at the top w level
    int tl;       // 1: tangents from ud/vd/phpd; 0: the seed's
    int which;    // the seeded input (DF_U, DF_V, DF_PH)
    size_t at;    // and its word
};

static __device__ __forceinline__ wreal df_tan(const DfSrc& s, int id, const wreal* d, size_t at)
{
    if (s.tl) return d[at];
    return (s.which == id && s.at == at) ? (wreal)1 : (wreal)0;
}

static __device__ __forceinline__ int df_ix(const DfSrc& s, int i)
{
    if (!s.bx) return df_per(i, s.nx);
    return i < 0 ? 0 : (i >= s.nx ? s.nx - 1 : i);
}
static __device__ __forceinline__ int df_iy(const DfSrc& s, int j)
{
    if (!s.by) return df_per(j, s.ny);
    return j < 0 ? 0 : (j >= s.ny ? s.ny - 1 : j);
}
static __device__ __forceinline__ int df_iu(const DfSrc& s, int i)
{
    if (!s.bx) return df_per(i, s.nx);
    return i < 0 ? 0 : (i > s.nx ? s.nx : i);
}
static __device__ __forceinline__ int df_jv(const DfSrc& s, int j)
{
    if (!s.by) return df_per(j, s.ny);
    return j < 0 ? 0 : (j > s.ny ? s.ny : j);
}

static __device__ __forceinline__ Dl df_php(const DfSrc& s, size_t at)
{
    return dl(s.php[at], df_tan(s, DF_PH, s.phpd, at));
}

// wrf_exact_z_at_w: (phb + ph') / g.
static __device__ Dl df_z(const DfSrc& s, int kw, int j, int i)
{
    int ii = df_ix(s, i), jj = df_iy(s, j);
    kw = kw < 0 ? 0 : (kw > s.nz ? s.nz : kw);
    size_t at = R3(kw, jj, ii, s.ny, s.nx);
    return (s.phb[at] + df_php(s, at)) / DF_G;
}

static __device__ Dl df_rdzw(const DfSrc& s, int k, int j, int i)
{
    k = k < 0 ? 0 : (k >= s.nz ? s.nz - 1 : k);
    return (wreal)1 / (df_z(s, k + 1, j, i) - df_z(s, k, j, i));
}

static __device__ Dl df_zx(const DfSrc& s, int kw, int j, int iface)
{
    if (s.bx && iface <= 0) return dl((wreal)0, (wreal)0);
    if (s.bx && iface >= s.nx) iface = s.nx - 1;
    // smag2d.cu wrf_zx: a periodic seam's top w level keeps WRF's cold zero
    // (RunConfig.diff_metrics_seam_slope off).
    if (s.seam_x && kw >= s.nz && df_per(iface, s.nx) == 0) return dl((wreal)0, (wreal)0);
    int jc = df_iy(s, j), ip = df_ix(s, iface), im = df_ix(s, iface - 1);
    kw = kw < 0 ? 0 : (kw > s.nz ? s.nz : kw);
    size_t a = R3(kw, jc, ip, s.ny, s.nx), b = R3(kw, jc, im, s.ny, s.nx);
    wreal base = wdiv(s.rdx * (s.phb[a] - s.phb[b]), DF_G);
    Dl pert = (s.rdx * (df_php(s, a) - df_php(s, b))) / DF_G;
    return base + pert;
}

static __device__ Dl df_zy(const DfSrc& s, int kw, int jface, int i)
{
    if (s.by && jface <= 0) return dl((wreal)0, (wreal)0);
    if (s.by && jface >= s.ny) jface = s.ny - 1;
    if (s.seam_y && kw >= s.nz && df_per(jface, s.ny) == 0) return dl((wreal)0, (wreal)0);
    int ic = df_ix(s, i), jp = df_iy(s, jface), jm = df_iy(s, jface - 1);
    kw = kw < 0 ? 0 : (kw > s.nz ? s.nz : kw);
    size_t a = R3(kw, jp, ic, s.ny, s.nx), b = R3(kw, jm, ic, s.ny, s.nx);
    wreal base = wdiv(s.rdy * (s.phb[a] - s.phb[b]), DF_G);
    Dl pert = (s.rdy * (df_php(s, a) - df_php(s, b))) / DF_G;
    return base + pert;
}

// u / msfu (isv 0) or v / msfv (isv 1) at (k, j, i) of its stagger.
static __device__ Dl df_hat(const DfSrc& s, int isv, int k, int j, int i)
{
    if (!isv) {
        int jj = df_iy(s, j), ii = df_iu(s, i);
        size_t at = R3(k, jj, ii, s.ny, s.nx + 1);
        return dl(s.u[at], df_tan(s, DF_U, s.ud, at)) / s.msfu[(size_t)jj * (s.nx + 1) + ii];
    }
    int jj = df_jv(s, j), ii = df_ix(s, i);
    size_t at = R3(k, jj, ii, s.ny + 1, s.nx);
    return dl(s.v[at], df_tan(s, DF_V, s.vd, at)) / s.msfv[(size_t)jj * s.nx + ii];
}

// wrf_u_w_xcenter / wrf_v_w_ycenter / wrf_u_w_corner / wrf_v_w_corner:
// the pair (j0, i0) + (j1, i1) of a wind's hat, at w level kw.
static __device__ Dl df_wpair(const DfSrc& s, int isv, int kw, int j0, int i0, int j1, int i1)
{
    if (kw <= 0)
        return (wreal)0.5 * (s.cf1 * df_hat(s, isv, 0, j0, i0) + s.cf2 * df_hat(s, isv, 1, j0, i0)
                             + s.cf3 * df_hat(s, isv, 2, j0, i0)
                             + s.cf1 * df_hat(s, isv, 0, j1, i1)
                             + s.cf2 * df_hat(s, isv, 1, j1, i1)
                             + s.cf3 * df_hat(s, isv, 2, j1, i1));
    int nz = s.nz;
    if (kw >= nz) {
        Dl pl = df_hat(s, isv, nz - 1, j0, i0) + df_hat(s, isv, nz - 1, j1, i1);
        Dl pp = df_hat(s, isv, nz - 2, j0, i0) + df_hat(s, isv, nz - 2, j1, i1);
        wreal cft2 = wdiv((wreal)-0.5 * s.dnw[nz - 1], s.dn[nz - 1]);
        return (wreal)0.5 * (((wreal)1 - cft2) * pl + cft2 * pp);
    }
    Dl pc = df_hat(s, isv, kw, j0, i0) + df_hat(s, isv, kw, j1, i1);
    Dl pb = df_hat(s, isv, kw - 1, j0, i0) + df_hat(s, isv, kw - 1, j1, i1);
    return (wreal)0.5 * (s.fnm[kw] * pc + s.fnp[kw] * pb);
}

static __device__ Dl df_d11(const DfSrc& s, int k, int j, int i)
{
    Dl tmpzx = (wreal)0.25 * (df_zx(s, k, j, i) + df_zx(s, k, j, i + 1) + df_zx(s, k + 1, j, i)
                              + df_zx(s, k + 1, j, i + 1));
    Dl slope = (df_wpair(s, 0, k + 1, j, i, j, i + 1) - df_wpair(s, 0, k, j, i, j, i + 1))
               * tmpzx * df_rdzw(s, k, j, i);
    wreal mm = s.msft[(size_t)df_iy(s, j) * s.nx + df_ix(s, i)];
    return (wreal)2 * mm * mm
           * (s.rdx * (df_hat(s, 0, k, j, i + 1) - df_hat(s, 0, k, j, i)) - slope);
}

static __device__ Dl df_d22(const DfSrc& s, int k, int j, int i)
{
    Dl tmpzy = (wreal)0.25 * (df_zy(s, k, j, i) + df_zy(s, k, j + 1, i) + df_zy(s, k + 1, j, i)
                              + df_zy(s, k + 1, j + 1, i));
    Dl slope = (df_wpair(s, 1, k + 1, j, i, j + 1, i) - df_wpair(s, 1, k, j, i, j + 1, i))
               * tmpzy * df_rdzw(s, k, j, i);
    wreal mm = s.msft[(size_t)df_iy(s, j) * s.nx + df_ix(s, i)];
    return (wreal)2 * mm * mm
           * (s.rdy * (df_hat(s, 1, k, j + 1, i) - df_hat(s, 1, k, j, i)) - slope);
}

static __device__ Dl df_d12(const DfSrc& s, int k, int j, int i)
{
    // cal_deform_and_div copies a bounded axis's outer tensor faces from
    // the adjacent interior ones.
    if (s.bx) i = i <= 0 ? (s.nx > 1 ? 1 : 0) : (i >= s.nx ? s.nx - 1 : i);
    if (s.by) j = j <= 0 ? (s.ny > 1 ? 1 : 0) : (j >= s.ny ? s.ny - 1 : j);
    int iu = df_iu(s, i), jm = df_iy(s, j - 1), jc = df_iy(s, j);
    int iv = df_ix(s, i), im = df_ix(s, i - 1), jv = df_jv(s, j);
    size_t wu = (size_t)s.nx + 1;
    wreal mm = (wreal)0.25 * (s.msfu[(size_t)jm * wu + iu] + s.msfu[(size_t)jc * wu + iu])
               * (s.msfv[(size_t)jv * s.nx + im] + s.msfv[(size_t)jv * s.nx + iv]);
    Dl rr = df_rdzw(s, k, j, i) + df_rdzw(s, k, j, i - 1) + df_rdzw(s, k, j - 1, i - 1)
            + df_rdzw(s, k, j - 1, i);
    Dl tmpzy = (wreal)0.25 * (df_zy(s, k, j, i - 1) + df_zy(s, k, j, i) + df_zy(s, k + 1, j, i - 1)
                              + df_zy(s, k + 1, j, i));
    Dl uslope = (df_wpair(s, 0, k + 1, j - 1, i, j, i) - df_wpair(s, 0, k, j - 1, i, j, i))
                * (wreal)0.25 * tmpzy * rr;
    Dl tmpzx = (wreal)0.25 * (df_zx(s, k, j - 1, i) + df_zx(s, k, j, i) + df_zx(s, k + 1, j - 1, i)
                              + df_zx(s, k + 1, j, i));
    Dl vslope = (df_wpair(s, 1, k + 1, j, i - 1, j, i) - df_wpair(s, 1, k, j, i - 1, j, i))
                * (wreal)0.25 * tmpzx
                * (df_rdzw(s, k, j, i) + df_rdzw(s, k, j - 1, i) + df_rdzw(s, k, j - 1, i - 1)
                   + df_rdzw(s, k, j, i - 1));
    return mm * (s.rdy * (df_hat(s, 0, k, j, i) - df_hat(s, 0, k, j - 1, i)) - uslope)
           + mm * (s.rdx * (df_hat(s, 1, k, j, i) - df_hat(s, 1, k, j, i - 1)) - vslope);
}

#define DF_PASSIVE                                                          \
    const wreal* __restrict__ phb, const wreal* __restrict__ msft,          \
    const wreal* __restrict__ msfu, const wreal* __restrict__ msfv,         \
    const wreal* __restrict__ fnm, const wreal* __restrict__ fnp,           \
    const wreal* __restrict__ dn, const wreal* __restrict__ dnw

#define DF_GEOM wreal rdx, wreal rdy, wreal cf1, wreal cf2, wreal cf3, int bx, int by, \
    int nz, int ny, int nx

static __device__ DfSrc df_src(const wreal* u, const wreal* ud, const wreal* v, const wreal* vd,
                               const wreal* php, const wreal* phpd, const wreal* phb,
                               const wreal* msft, const wreal* msfu, const wreal* msfv,
                               const wreal* fnm, const wreal* fnp, const wreal* dn,
                               const wreal* dnw, wreal rdx, wreal rdy, wreal cf1, wreal cf2,
                               wreal cf3, int bx, int by, int nz, int ny, int nx, int tl)
{
    DfSrc s;
    s.u = u; s.ud = ud; s.v = v; s.vd = vd; s.php = php; s.phpd = phpd; s.phb = phb;
    s.msft = msft; s.msfu = msfu; s.msfv = msfv; s.fnm = fnm; s.fnp = fnp; s.dn = dn;
    s.dnw = dnw;
    s.rdx = rdx; s.rdy = rdy; s.cf1 = cf1; s.cf2 = cf2; s.cf3 = cf3;
    // the primal's boundary flags (smag2d.cu wrf_grid): 2 is a periodic
    // axis with WRF's cold seam slope at the top w level
    s.nz = nz; s.ny = ny; s.nx = nx; s.bx = bx & 1; s.by = by & 1;
    s.seam_x = bx == 2; s.seam_y = by == 2;
    s.tl = tl; s.which = 0; s.at = 0;
    return s;
}

// TL: one thread per mass point; d11, d22 and d12 are overwritten.
extern "C" __global__ void df_tl(wreal* __restrict__ d11, wreal* __restrict__ d22,
                                 wreal* __restrict__ d12, wreal* __restrict__ d11d,
                                 wreal* __restrict__ d22d, wreal* __restrict__ d12d,
                                 const wreal* __restrict__ u, const wreal* __restrict__ ud,
                                 const wreal* __restrict__ v, const wreal* __restrict__ vd,
                                 const wreal* __restrict__ php, const wreal* __restrict__ phpd,
                                 DF_PASSIVE, int nl, DF_GEOM)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)nz * ny * nx;
    if (t >= n) return;
    int i = (int)(t % nx), j = (int)((t / nx) % ny), k = (int)(t / ((size_t)nx * ny));
    DfSrc s = df_src(u, ud, v, vd, php, phpd, phb, msft, msfu, msfv, fnm, fnp, dn, dnw, rdx,
                     rdy, cf1, cf2, cf3, bx, by, nz, ny, nx, 1);
    Dl a = df_d11(s, k, j, i), b = df_d22(s, k, j, i), c = df_d12(s, k, j, i);
    if (nl) {
        d11[t] = a.v;
        d22[t] = b.v;
        d12[t] = c.v;
    }
    if (WRFPLUS_TL) {
        d11d[t] = a.d;
        d22d[t] = b.d;
        d12d[t] = c.d;
    }
}

// AD: one thread per input word (u faces, then v faces, then ph' levels).
// The outputs that can read a word lie within one column and row of it
// (a bounded axis's copied outer d12 face included): defor11 at (j, i)
// reads u faces i and i+1 and ph' columns i-1..i+1 of row j only, defor22
// the same for v and rows, defor12 u rows j-1..j at face i, v columns
// i-1..i at face j and ph' rows and columns j-1..j, i-1..i.  A wind word
// at level L is read at the levels L-1..L+1, by level 0 when L <= 2 (the
// bottom extrapolation reads levels 0-2) and by level nz-1 when
// L >= nz-2 (the lid's reads nz-2..nz-1); a ph' word at w level L by
// levels L-1 and L.  df_cand drops repeats.  Within that window each
// term is gathered only from the outputs that read the word (tested on
// the output's coordinate, so a short periodic axis whose images repeat
// still finds it): defor11 from columns i-1..i for a u face, defor22
// from rows j-1..j for a v face; defor12 from face i (u) or j (v) and
// the next row (u), column (v) or both (ph'), plus a bounded axis's
// copied outer face 0 for index 1.
extern "C" __global__ void df_ad(const wreal* __restrict__ a11, const wreal* __restrict__ a22,
                                 const wreal* __restrict__ a12, const wreal* __restrict__ u,
                                 const wreal* __restrict__ v, const wreal* __restrict__ php,
                                 wreal* __restrict__ au, wreal* __restrict__ av,
                                 wreal* __restrict__ aph, DF_PASSIVE, DF_GEOM)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t nu = (size_t)nz * ny * (nx + 1), nv = (size_t)nz * (ny + 1) * nx;
    size_t nw = (size_t)(nz + 1) * ny * nx;
    if (t >= nu + nv + nw) return;
    DfSrc s = df_src(u, u, v, v, php, php, phb, msft, msfu, msfv, fnm, fnp, dn, dnw, rdx, rdy,
                     cf1, cf2, cf3, bx, by, nz, ny, nx, 0);
    bx &= 1;
    by &= 1;
    int k, j, i, w, h;
    wreal* out;
    if (t < nu) {
        w = nx + 1; h = ny;
        s.which = DF_U; s.at = t; out = au;
    } else if (t < nu + nv) {
        t -= nu;
        w = nx; h = ny + 1;
        s.which = DF_V; s.at = t; out = av;
    } else {
        t -= nu + nv;
        w = nx; h = ny;
        s.which = DF_PH; s.at = t; out = aph;
    }
    i = (int)(t % w); j = (int)((t / w) % h); k = (int)(t / ((size_t)w * h));
    // a face index nx (ny) of a periodic axis is never read: its image is
    // 0, so its adjoint gains nothing.
    if ((s.which == DF_U && !bx && i == nx) || (s.which == DF_V && !by && j == ny)) return;
    int ph = s.which == DF_PH;
    // the lid's reads (nz-2..nz-1) already lie in k..k+1; the bottom
    // extrapolation adds level 0 only for k == 2
    int levels[4] = {k - 1, k, ph ? -1 : k + 1, (!ph && k == 2) ? 0 : -1};
    // the previous and next images of i and j on their axes
    int ip = bx ? i - 1 : df_per(i - 1, nx), inx = bx ? i + 1 : df_per(i + 1, nx);
    int jp = by ? j - 1 : df_per(j - 1, ny), jnx = by ? j + 1 : df_per(j + 1, ny);
    wreal acc = (wreal)0;
    for (int a = 0; a < 4; ++a) {
        int ko = levels[a];
        if (ko < 0 || ko >= nz) continue;
        bool seen = false;
        for (int b = 0; b < a; ++b) seen = seen || levels[b] == ko;
        if (seen) continue;
        for (int oj = -1; oj <= 1; ++oj) {
            int jo = df_cand(j, oj, -1, ny, by);
            if (jo < 0) continue;
            for (int oi = -1; oi <= 1; ++oi) {
                int io = df_cand(i, oi, -1, nx, bx);
                if (io < 0) continue;
                size_t o = R3(ko, jo, io, ny, nx);
                // defor11 reads no v and only row j, defor22 no u and
                // only column i
                if (a11[o] != (wreal)0 && s.which != DF_V && jo == j
                    && (s.which == DF_PH || io == i || io == ip))
                    acc = acc + a11[o] * df_d11(s, ko, jo, io).d;
                if (a22[o] != (wreal)0 && s.which != DF_U && io == i
                    && (s.which == DF_PH || jo == j || jo == jp))
                    acc = acc + a22[o] * df_d22(s, ko, jo, io).d;
                bool x12 = io == i || (bx && i == 1 && io == 0)
                           || (s.which != DF_U && io == inx);
                bool y12 = jo == j || (by && j == 1 && jo == 0)
                           || (s.which != DF_V && jo == jnx);
                if (a12[o] != (wreal)0 && x12 && y12)
                    acc = acc + a12[o] * df_d12(s, ko, jo, io).d;
            }
        }
    }
    out[t] = out[t] + acc;
}

// The overwritten outputs' adjoints are consumed.
extern "C" __global__ void df_zero3(wreal* __restrict__ a, wreal* __restrict__ b,
                                    wreal* __restrict__ c, int n)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)n) return;
    a[t] = (wreal)0;
    b[t] = (wreal)0;
    c[t] = (wreal)0;
}

// ====================================================== smag2d_km (diff_opt 1)

#define KM_D11 1
#define KM_D22 2
#define KM_D12 3

struct KmSrc {
    const wreal* d11;
    const wreal* d11d;
    const wreal* d22;
    const wreal* d22d;
    const wreal* d12;
    const wreal* d12d;
    const wreal* msft;
    wreal dx, dy, cs;
    int nz, ny, nx, bx, by;
    int tl, which;
    size_t at;
};

static __device__ __forceinline__ Dl km_in(const KmSrc& s, int id, const wreal* a,
                                           const wreal* ad, size_t at)
{
    wreal d = s.tl ? ad[at] : ((s.which == id && s.at == at) ? (wreal)1 : (wreal)0);
    return dl(a[at], d);
}

// wrf_diff_opt1_km4 at (k, j, i) (then wrf_smag_km_bc's zero outer row).
// WRFPLUS's derivatives: sqrt' = 0 where def2 = 0, and none where the
// coefficient takes the 10 * mlen cap.
static __device__ Dl km_point(const KmSrc& s, int k, int j, int i)
{
    if ((s.bx && (i == 0 || i == s.nx - 1)) || (s.by && (j == 0 || j == s.ny - 1)))
        return dl((wreal)0, (wreal)0);
    int ip = s.bx ? (i + 1 > s.nx - 1 ? s.nx - 1 : i + 1) : df_per(i + 1, s.nx);
    int jp = s.by ? (j + 1 > s.ny - 1 ? s.ny - 1 : j + 1) : df_per(j + 1, s.ny);
    size_t idx = R3(k, j, i, s.ny, s.nx);
    Dl d = km_in(s, KM_D11, s.d11, s.d11d, idx) - km_in(s, KM_D22, s.d22, s.d22d, idx);
    Dl cross = (wreal)0.25 * (((km_in(s, KM_D12, s.d12, s.d12d, idx)
                                + km_in(s, KM_D12, s.d12, s.d12d, R3(k, jp, i, s.ny, s.nx)))
                               + km_in(s, KM_D12, s.d12, s.d12d, R3(k, j, ip, s.ny, s.nx)))
                              + km_in(s, KM_D12, s.d12, s.d12d, R3(k, jp, ip, s.ny, s.nx)));
    Dl def2 = (wreal)0.25 * (d * d) + cross * cross;
    wreal root = wsqrt(def2.v);
    Dl strain = dl(root, def2.v == (wreal)0 ? (wreal)0 : wdiv(def2.d, (wreal)2 * root));
    wreal map = s.msft[(size_t)j * s.nx + i];
    wreal length = wsqrt(wdiv(wdiv(s.dx, map) * s.dy, map));
    Dl value = s.cs * s.cs * length * length * strain;
    wreal cap = (wreal)10 * length;
    if (value.v > cap) return dl(cap, (wreal)0);
    return value;
}

#define KM_GEOM wreal dx, wreal dy, wreal cs, wreal pr, int bx, int by, int nz, int ny, int nx

static __device__ KmSrc km_src(const wreal* d11, const wreal* d11d, const wreal* d22,
                               const wreal* d22d, const wreal* d12, const wreal* d12d,
                               const wreal* msft, wreal dx, wreal dy, wreal cs, int bx, int by,
                               int nz, int ny, int nx, int tl)
{
    KmSrc s;
    s.d11 = d11; s.d11d = d11d; s.d22 = d22; s.d22d = d22d; s.d12 = d12; s.d12d = d12d;
    s.msft = msft; s.dx = dx; s.dy = dy; s.cs = cs;
    s.nz = nz; s.ny = ny; s.nx = nx; s.bx = bx; s.by = by;
    s.tl = tl; s.which = 0; s.at = 0;
    return s;
}

extern "C" __global__ void km_tl(wreal* __restrict__ km, wreal* __restrict__ kh,
                                 wreal* __restrict__ kmd, wreal* __restrict__ khd,
                                 const wreal* __restrict__ d11, const wreal* __restrict__ d11d,
                                 const wreal* __restrict__ d22, const wreal* __restrict__ d22d,
                                 const wreal* __restrict__ d12, const wreal* __restrict__ d12d,
                                 const wreal* __restrict__ msft, int nl, KM_GEOM)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nz * ny * nx) return;
    int i = (int)(t % nx), j = (int)((t / nx) % ny), k = (int)(t / ((size_t)nx * ny));
    KmSrc s = km_src(d11, d11d, d22, d22d, d12, d12d, msft, dx, dy, cs, bx, by, nz, ny, nx, 1);
    Dl m = km_point(s, k, j, i);
    Dl h = m / pr;
    if (nl) {
        km[t] = m.v;
        kh[t] = h.v;
    }
    if (WRFPLUS_TL) {
        kmd[t] = m.d;
        khd[t] = h.d;
    }
}

// AD: one thread per mass point adds into the d11, d22 (read by its own
// coefficient) and d12 (read by the coefficients at j-1..j, i-1..i)
// adjoints.
extern "C" __global__ void km_ad(const wreal* __restrict__ akm, const wreal* __restrict__ akh,
                                 const wreal* __restrict__ d11, const wreal* __restrict__ d22,
                                 const wreal* __restrict__ d12, wreal* __restrict__ a11,
                                 wreal* __restrict__ a22, wreal* __restrict__ a12,
                                 const wreal* __restrict__ msft, KM_GEOM)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nz * ny * nx) return;
    int i = (int)(t % nx), j = (int)((t / nx) % ny), k = (int)(t / ((size_t)nx * ny));
    KmSrc s = km_src(d11, d11, d22, d22, d12, d12, msft, dx, dy, cs, bx, by, nz, ny, nx, 0);
    s.at = t;
    s.which = KM_D11;
    Dl m = km_point(s, k, j, i);
    a11[t] = a11[t] + (akm[t] * m.d + akh[t] * wdiv(m.d, pr));
    s.which = KM_D22;
    m = km_point(s, k, j, i);
    a22[t] = a22[t] + (akm[t] * m.d + akh[t] * wdiv(m.d, pr));
    s.which = KM_D12;
    wreal acc = (wreal)0;
    for (int oj = -1; oj <= 0; ++oj) {
        int jo = df_cand(j, oj, -1, ny, by);
        if (jo < 0) continue;
        for (int oi = -1; oi <= 0; ++oi) {
            int io = df_cand(i, oi, -1, nx, bx);
            if (io < 0) continue;
            size_t o = R3(k, jo, io, ny, nx);
            m = km_point(s, k, jo, io);
            acc = acc + (akm[o] * m.d + akh[o] * wdiv(m.d, pr));
        }
    }
    a12[t] = a12[t] + acc;
}

extern "C" __global__ void km_zero2(wreal* __restrict__ a, wreal* __restrict__ b, int n)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)n) return;
    a[t] = (wreal)0;
    b[t] = (wreal)0;
}

// ============================================ horizontal_diffusion (rows)
//
// woof wrf_diff_opt1_horizontal (diff_opt1.cu): WRF horizontal_diffusion
// for the u (stag 1), v (2), w (3) and scalar (0) rows and
// horizontal_diffusion_3dmp (0 with perturb: theta minus its initial
// field), coordinate-surface fluxes with the hybrid mass c1 mut + c2.
// The tendency accumulates (tendency += ...), so its adjoint passes
// through and is not consumed.  Active: the field, the coefficient
// (xkmh or xkhh) and mut; passive: c1, c2, the base field and the map
// factors.  WRFPLUS 4.7.1's G_/A_HORIZONTAL_DIFFUSION(_3DMP) take mut as
// the mass (no c1/c2), so the oracle cases use c1 = 1, c2 = 0.

#define HD_F 1
#define HD_K 2
#define HD_MU 3

static __device__ __forceinline__ Dl operator+(Dl a, wreal b) { return dl(a.v + b, a.d); }

struct HdSrc {
    const wreal* f;
    const wreal* fd;
    const wreal* km;
    const wreal* kmd;
    const wreal* mu;
    const wreal* mud;
    const wreal* c1;
    const wreal* c2;
    const wreal* base;
    const wreal* msft;
    const wreal* msfu;
    const wreal* msfv;
    wreal rdx, rdy;
    int nz, ny, nx, stag, bx, by, perturb;
    int tl, which;
    size_t at;
};

static __device__ __forceinline__ Dl hd_in(const HdSrc& s, int id, const wreal* a,
                                           const wreal* ad, size_t at)
{
    wreal d = s.tl ? ad[at] : ((s.which == id && s.at == at) ? (wreal)1 : (wreal)0);
    return dl(a[at], d);
}

// d1index: clamped on a bounded axis, periodic otherwise
static __device__ __forceinline__ int hd_ix(int n, int size, int bounded)
{
    return bounded ? max(0, min(n, size - 1)) : df_per(n, size);
}

static __device__ __forceinline__ Dl hd_mass(const HdSrc& s, int k, int j, int i)
{
    size_t at = (size_t)hd_ix(j, s.ny, s.by) * s.nx + hd_ix(i, s.nx, s.bx);
    return s.c1[k] * hd_in(s, HD_MU, s.mu, s.mud, at) + s.c2[k];
}

static __device__ __forceinline__ Dl hd_k(const HdSrc& s, int k, int j, int i)
{
    size_t at = R3(max(0, min(k, s.nz - 1)), hd_ix(j, s.ny, s.by), hd_ix(i, s.nx, s.bx),
                   s.ny, s.nx);
    return hd_in(s, HD_K, s.km, s.kmd, at);
}

static __device__ __forceinline__ Dl hd_f(const HdSrc& s, int k, int j, int i)
{
    int ii = s.stag == 1 ? (s.bx ? max(0, min(i, s.nx)) : df_per(i, s.nx)) : hd_ix(i, s.nx, s.bx);
    int jj = s.stag == 2 ? (s.by ? max(0, min(j, s.ny)) : df_per(j, s.ny)) : hd_ix(j, s.ny, s.by);
    return hd_in(s, HD_F, s.f, s.fd, R3(k, jj, ii, s.ny + (s.stag == 2), s.nx + (s.stag == 1)));
}

static __device__ __forceinline__ wreal hd_b(const HdSrc& s, int k, int j, int i)
{
    return s.base[R3(k, hd_ix(j, s.ny, s.by), hd_ix(i, s.nx, s.bx), s.ny, s.nx)];
}

static __device__ __forceinline__ wreal hd_map(const HdSrc& s, int stagger, int j, int i)
{
    int width = s.nx + (stagger == 1);
    int ii = stagger == 1 && s.bx ? max(0, min(i, s.nx)) : hd_ix(i, s.nx, s.bx);
    int jj = stagger == 2 && s.by ? max(0, min(j, s.ny)) : hd_ix(j, s.ny, s.by);
    return (stagger == 1 ? s.msfu : stagger == 2 ? s.msfv : s.msft)[(size_t)jj * width + ii];
}

static __device__ __forceinline__ Dl hd_four(Dl a, Dl b, Dl c, Dl d)
{
    return (wreal)0.25 * (((a + b) + c) + d);
}

static __device__ __forceinline__ Dl hd_delta(const HdSrc& s, int k, int j1, int i1, int j0,
                                              int i0)
{
    Dl v = hd_f(s, k, j1, i1) - hd_f(s, k, j0, i0);
    if (s.perturb) v = dl((v.v - hd_b(s, k, j1, i1)) + hd_b(s, k, j0, i0), v.d);
    return v;
}

// The primal's increment at output (k, j, i), 0 off its loop.
static __device__ Dl hd_point(const HdSrc& s, int k, int j, int i)
{
    const int st = s.stag;
    if (st == 3 && (k == 0 || k == s.nz)) return dl((wreal)0, (wreal)0);
    if (s.bx && (i == 0 || i >= s.nx - (st != 1))) return dl((wreal)0, (wreal)0);
    if (s.by && (j == 0 || j >= s.ny - (st != 2))) return dl((wreal)0, (wreal)0);
    Dl xm, xp, ym, yp;
    if (st == 1) {
        xm = (hd_mass(s, k, j, i - 1) * hd_k(s, k, j, i - 1)) * s.rdx;
        xp = (hd_mass(s, k, j, i) * hd_k(s, k, j, i)) * s.rdx;
        ym = (hd_four(hd_mass(s, k, j, i), hd_mass(s, k, j - 1, i), hd_mass(s, k, j - 1, i - 1),
                      hd_mass(s, k, j, i - 1))
              * hd_four(hd_k(s, k, j, i), hd_k(s, k, j - 1, i), hd_k(s, k, j - 1, i - 1),
                        hd_k(s, k, j, i - 1))) * s.rdy;
        yp = (hd_four(hd_mass(s, k, j, i), hd_mass(s, k, j + 1, i), hd_mass(s, k, j + 1, i - 1),
                      hd_mass(s, k, j, i - 1))
              * hd_four(hd_k(s, k, j, i), hd_k(s, k, j + 1, i), hd_k(s, k, j + 1, i - 1),
                        hd_k(s, k, j, i - 1))) * s.rdy;
    } else if (st == 2) {
        xm = (hd_four(hd_mass(s, k, j, i), hd_mass(s, k, j - 1, i), hd_mass(s, k, j - 1, i - 1),
                      hd_mass(s, k, j, i - 1))
              * hd_four(hd_k(s, k, j, i), hd_k(s, k, j - 1, i), hd_k(s, k, j - 1, i - 1),
                        hd_k(s, k, j, i - 1))) * s.rdx;
        xp = (hd_four(hd_mass(s, k, j, i), hd_mass(s, k, j - 1, i), hd_mass(s, k, j - 1, i + 1),
                      hd_mass(s, k, j, i + 1))
              * hd_four(hd_k(s, k, j, i), hd_k(s, k, j - 1, i), hd_k(s, k, j - 1, i + 1),
                        hd_k(s, k, j, i + 1))) * s.rdx;
        // WRF v4.7.1's coordinate-v meridional row has no dry mass
        ym = hd_k(s, k, j - 1, i) * s.rdy;
        yp = hd_k(s, k, j, i) * s.rdy;
    } else if (st == 3) {
        xm = (hd_four(hd_mass(s, k, j, i), hd_mass(s, k, j, i - 1), hd_mass(s, k, j, i),
                      hd_mass(s, k, j, i - 1))
              * hd_four(hd_k(s, k, j, i), hd_k(s, k, j, i - 1), hd_k(s, k - 1, j, i),
                        hd_k(s, k - 1, j, i - 1))) * s.rdx;
        xp = (hd_four(hd_mass(s, k, j, i + 1), hd_mass(s, k, j, i), hd_mass(s, k, j, i + 1),
                      hd_mass(s, k, j, i))
              * hd_four(hd_k(s, k, j, i + 1), hd_k(s, k, j, i), hd_k(s, k - 1, j, i + 1),
                        hd_k(s, k - 1, j, i))) * s.rdx;
        wreal m0 = hd_map(s, 2, j, i), m1 = hd_map(s, 2, j + 1, i);
        ym = ((m0 * wdiv((wreal)1, m0))
              * hd_four(hd_mass(s, k, j, i), hd_mass(s, k, j - 1, i), hd_mass(s, k, j, i),
                        hd_mass(s, k, j - 1, i)))
             * hd_four(hd_k(s, k, j, i), hd_k(s, k, j - 1, i), hd_k(s, k - 1, j, i),
                       hd_k(s, k - 1, j - 1, i));
        ym = ym * s.rdy;
        yp = ((m1 * wdiv((wreal)1, m1))
              * hd_four(hd_mass(s, k, j + 1, i), hd_mass(s, k, j, i), hd_mass(s, k, j + 1, i),
                        hd_mass(s, k, j, i)))
             * hd_four(hd_k(s, k, j + 1, i), hd_k(s, k, j, i), hd_k(s, k - 1, j + 1, i),
                       hd_k(s, k - 1, j, i));
        yp = yp * s.rdy;
    } else {
        xm = ((((wreal)0.5 * (hd_k(s, k, j, i) + hd_k(s, k, j, i - 1))) * (wreal)0.5)
              * (hd_mass(s, k, j, i) + hd_mass(s, k, j, i - 1))) * s.rdx;
        xp = ((((wreal)0.5 * (hd_k(s, k, j, i + 1) + hd_k(s, k, j, i))) * (wreal)0.5)
              * (hd_mass(s, k, j, i + 1) + hd_mass(s, k, j, i))) * s.rdx;
        wreal m0 = hd_map(s, 2, j, i), m1 = hd_map(s, 2, j + 1, i);
        ym = (((((m0 * wdiv((wreal)1, m0)) * (wreal)0.5)
                * (hd_k(s, k, j, i) + hd_k(s, k, j - 1, i))) * (wreal)0.5)
              * (hd_mass(s, k, j, i) + hd_mass(s, k, j - 1, i)));
        ym = ym * s.rdy;
        yp = (((((m1 * wdiv((wreal)1, m1)) * (wreal)0.5)
                * (hd_k(s, k, j + 1, i) + hd_k(s, k, j, i))) * (wreal)0.5)
              * (hd_mass(s, k, j + 1, i) + hd_mass(s, k, j, i)));
        yp = yp * s.rdy;
    }
    wreal map = hd_map(s, st == 3 ? 0 : st, j, i), map2 = map * map;
    Dl tx = (map2 * s.rdx) * (xp * hd_delta(s, k, j, i + 1, j, i)
                              - xm * hd_delta(s, k, j, i, j, i - 1));
    Dl ty = (map2 * s.rdy) * (yp * hd_delta(s, k, j + 1, i, j, i)
                              - ym * hd_delta(s, k, j, i, j - 1, i));
    return tx + ty;
}

#define HD_PASSIVE const wreal* __restrict__ c1, const wreal* __restrict__ c2, \
    const wreal* __restrict__ base, const wreal* __restrict__ msft,             \
    const wreal* __restrict__ msfu, const wreal* __restrict__ msfv

#define HD_GEOM wreal rdx, wreal rdy, int stag, int perturb, int bx, int by, int nz, int ny, \
    int nx

static __device__ HdSrc hd_src(const wreal* f, const wreal* fd, const wreal* km,
                               const wreal* kmd, const wreal* mu, const wreal* mud,
                               const wreal* c1, const wreal* c2, const wreal* base,
                               const wreal* msft, const wreal* msfu, const wreal* msfv,
                               wreal rdx, wreal rdy, int stag, int perturb, int bx, int by,
                               int nz, int ny, int nx, int tl)
{
    HdSrc s;
    s.f = f; s.fd = fd; s.km = km; s.kmd = kmd; s.mu = mu; s.mud = mud;
    s.c1 = c1; s.c2 = c2; s.base = base; s.msft = msft; s.msfu = msfu; s.msfv = msfv;
    s.rdx = rdx; s.rdy = rdy;
    s.nz = nz; s.ny = ny; s.nx = nx; s.stag = stag; s.bx = bx; s.by = by;
    s.perturb = perturb; s.tl = tl; s.which = 0; s.at = 0;
    return s;
}

// TL: one thread per point of the row; the tendency accumulates.
extern "C" __global__ void hd_tl(wreal* __restrict__ tend, wreal* __restrict__ tendd,
                                 const wreal* __restrict__ f, const wreal* __restrict__ fd,
                                 const wreal* __restrict__ km, const wreal* __restrict__ kmd,
                                 const wreal* __restrict__ mu, const wreal* __restrict__ mud,
                                 HD_PASSIVE, int nl, HD_GEOM)
{
    int nxs = nx + (stag == 1), nys = ny + (stag == 2), nzs = nz + (stag == 3);
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nzs * nys * nxs) return;
    int i = (int)(t % nxs), j = (int)((t / nxs) % nys), k = (int)(t / ((size_t)nxs * nys));
    HdSrc s = hd_src(f, fd, km, kmd, mu, mud, c1, c2, base, msft, msfu, msfv, rdx, rdy, stag,
                     perturb, bx, by, nz, ny, nx, 1);
    if (stag == 3 && (k == 0 || k == nz)) return;
    if (bx && (i == 0 || i >= nx - (stag != 1))) return;
    if (by && (j == 0 || j >= ny - (stag != 2))) return;
    Dl r = hd_point(s, k, j, i);
    if (nl) tend[t] = tend[t] + r.v;
    if (WRFPLUS_TL) tendd[t] = tendd[t] + r.d;
}

// The distinct row points within one point of coordinate c on an axis
// of n mass points (n + stag row points): on a periodic axis the images
// mod n, and the staggered end point n with image 0.
static __device__ int hd_axis(int c, int n, int bounded, int stag, int* out)
{
    int m = 0;
    for (int off = -1; off <= 1; ++off) {
        int x = c + off;
        if (bounded) {
            if (x >= 0 && x < n + stag) out[m++] = x;
            continue;
        }
        x = df_per(x, n);
        bool seen = false;
        for (int q = 0; q < m; ++q) seen = seen || out[q] == x;
        if (seen) continue;
        out[m++] = x;
        if (stag && x == 0) out[m++] = n;
    }
    return m;
}

// AD: one thread per input word (field, then coefficient, then mut).  A
// field word is read at its own level, a coefficient word at its own and
// (w row) the next, a mut word at every level; horizontally within one
// point.  The tendency adjoint is read, not consumed.
extern "C" __global__ void hd_ad(const wreal* __restrict__ atend, const wreal* __restrict__ f,
                                 const wreal* __restrict__ km, const wreal* __restrict__ mu,
                                 wreal* __restrict__ af, wreal* __restrict__ akm,
                                 wreal* __restrict__ amu, HD_PASSIVE, HD_GEOM)
{
    int nxs = nx + (stag == 1), nys = ny + (stag == 2), nzs = nz + (stag == 3);
    size_t nf = (size_t)nzs * nys * nxs, nk = (size_t)nz * ny * nx, nm = (size_t)ny * nx;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nf + nk + nm) return;
    HdSrc s = hd_src(f, f, km, km, mu, mu, c1, c2, base, msft, msfu, msfv, rdx, rdy, stag,
                     perturb, bx, by, nz, ny, nx, 0);
    int k, j, i, klo, khi;
    wreal* out;
    if (t < nf) {
        i = (int)(t % nxs); j = (int)((t / nxs) % nys); k = (int)(t / ((size_t)nxs * nys));
        // a periodic staggered end point is never read (its image is 0)
        if ((stag == 1 && !bx && i == nx) || (stag == 2 && !by && j == ny)) return;
        s.which = HD_F; s.at = t; out = af;
        klo = khi = k;
    } else if (t < nf + nk) {
        t -= nf;
        i = (int)(t % nx); j = (int)((t / nx) % ny); k = (int)(t / ((size_t)nx * ny));
        s.which = HD_K; s.at = t; out = akm;
        klo = k; khi = stag == 3 ? k + 1 : k;
    } else {
        t -= nf + nk;
        i = (int)(t % nx); j = (int)(t / nx); k = 0;
        s.which = HD_MU; s.at = t; out = amu;
        klo = 0; khi = nzs - 1;
    }
    // a field word's own axis may be staggered; the coefficient's and
    // mut's are mass points read by the row's points
    int js[4], is[4];
    int mj = hd_axis(j, ny, by, stag == 2, js);
    int mi = hd_axis(i, nx, bx, stag == 1, is);
    wreal acc = (wreal)0;
    for (int ko = klo; ko <= khi && ko < nzs; ++ko)
        for (int b = 0; b < mj; ++b)
            for (int a = 0; a < mi; ++a) {
                size_t o = R3(ko, js[b], is[a], nys, nxs);
                if (atend[o] != (wreal)0) acc = acc + atend[o] * hd_point(s, ko, js[b], is[a]).d;
            }
    out[t] = out[t] + acc;
}

// ======================================= diff_opt 2 rows (horizontal_diffusion_2)
//
// WRF horizontal_diffusion_u_2 / _v_2 / _w_2 / _s (module_diffusion_em.F,
// the density-weighted stresses and fluxes of WRF 4.7.1) on woof's strict
// wrf_smag_hd_u / wrf_smag_hd_v / wrf_smag_hd_w and wrf_smag_flux_s +
// wrf_smag_hd_s (smag2d.cu, the GPUWM_WRF_EXACT_C_DIFFUSION statements;
// smag2d_uv_faces.cu and smag2d_w_faces.cu give the same words).
//
// The primal recomputes every stress and metric at each point that reads
// it.  Here the forward is staged through stored intermediates, each the
// word the primal forms inline (so the float base state is unchanged):
//
//   metrics  zx (w levels, u faces), zy (w levels, v faces), rdzw (mass)
//            and rdz (w levels) of ph' (compute_diff_metrics);
//   rho      1/alt, times 1 + qv when moist;
//   tuv      tau11 or tau22 (mass) and tau12 (corners) of rho, K and the
//            deformation (u and v rows);
//   tw       tau13 (w levels, u faces) and tau23 (w levels, v faces) of
//            rho, K, u, v, w and the metrics (w row, defor13/23 inline);
//   hs       the scalar fluxes H1 (u faces) and H2 (v faces) of the field,
//            rho, K and the metrics (scalar and theta rows);
//   row      the tendency increment of the stresses or fluxes and the
//            metrics.
//
// Each stage's TL evaluates its outputs on value/tangent pairs; its AD
// gathers, per input word, a_out * d out / d word over the outputs of the
// stage whose stencil can read the word (the same seeded forward as the
// diff_opt 1 kernels above).  The stages' adjoints run in reverse order
// through the stored base intermediates, so no stage recomputes another.

#define H2_PHP 1
#define H2_ALT 2
#define H2_QV 3
#define H2_RHO 4
#define H2_K 5
#define H2_DA 6
#define H2_D12 7
#define H2_U 8
#define H2_V 9
#define H2_W 10
#define H2_F 11
#define H2_ZX 12
#define H2_ZY 13
#define H2_RZ 14
#define H2_S1 15
#define H2_S2 16

#define H2O_ZX 1
#define H2O_ZY 2
#define H2O_RDZW 3
#define H2O_RDZ 4
#define H2O_RHO 5
#define H2O_TA 6
#define H2O_T12 7
#define H2O_T13 8
#define H2O_T23 9
#define H2O_H1 10
#define H2O_H2 11
#define H2O_TEND 12

#define H2S_MET 1
#define H2S_RHO 2
#define H2S_TUV 3
#define H2S_TW 4
#define H2S_HS 5
#define H2S_ROW 6

// rows: 1 u, 2 v, 3 w, 0 a scalar (theta is strict thp = theta - 300)

static __device__ __forceinline__ Dl operator-(Dl a) { return dl(-a.v, -a.d); }
// a / b, both active: the value is the primal's division.
static __device__ __forceinline__ Dl operator/(Dl a, Dl b)
{
    wreal v = wdiv(a.v, b.v);
    return dl(v, wdiv(a.d - v * b.d, b.v));
}

struct H2Src {
    const wreal* php;
    const wreal* phpd;
    const wreal* phb;
    const wreal* alt;
    const wreal* altd;
    const wreal* qv;
    const wreal* qvd;
    const wreal* rho;
    const wreal* rhod;
    const wreal* km;
    const wreal* kmd;
    const wreal* da;
    const wreal* dad;
    const wreal* d12;
    const wreal* d12d;
    const wreal* u;
    const wreal* ud;
    const wreal* v;
    const wreal* vd;
    const wreal* w;
    const wreal* wd;
    const wreal* f;
    const wreal* fd;
    const wreal* zx;
    const wreal* zxd;
    const wreal* zy;
    const wreal* zyd;
    const wreal* rz;      // rdzw (tuv/hs, u/v/scalar rows) or rdz (tw, w row)
    const wreal* rzd;
    const wreal* s1;      // the row's stresses or fluxes
    const wreal* s1d;
    const wreal* s2;
    const wreal* s2d;
    const wreal* msft;
    const wreal* msfu;
    const wreal* msfv;
    const wreal* fnm;
    const wreal* fnp;
    const wreal* dn;
    const wreal* dnw;
    wreal rdx, rdy, cf1, cf2, cf3;
    int nz, ny, nx, bx, by, row, moist;
    int mode;     // 1: tangents from the *d arrays; 0: the seed's; 2: none
    int which;    // the seeded input (H2_*)
    size_t at;    // and its word
};

static __device__ H2Src h2_src(wreal rdx, wreal rdy, wreal cf1, wreal cf2, wreal cf3, int bx,
                               int by, int nz, int ny, int nx, int row, int moist, int mode)
{
    H2Src s;
    s.php = s.phpd = s.phb = nullptr;
    s.alt = s.altd = s.qv = s.qvd = s.rho = s.rhod = s.km = s.kmd = nullptr;
    s.da = s.dad = s.d12 = s.d12d = nullptr;
    s.u = s.ud = s.v = s.vd = s.w = s.wd = s.f = s.fd = nullptr;
    s.zx = s.zxd = s.zy = s.zyd = s.rz = s.rzd = nullptr;
    s.s1 = s.s1d = s.s2 = s.s2d = nullptr;
    s.msft = s.msfu = s.msfv = s.fnm = s.fnp = s.dn = s.dnw = nullptr;
    s.rdx = rdx; s.rdy = rdy; s.cf1 = cf1; s.cf2 = cf2; s.cf3 = cf3;
    s.bx = bx; s.by = by; s.nz = nz; s.ny = ny; s.nx = nx; s.row = row; s.moist = moist;
    s.mode = mode; s.which = 0; s.at = 0;
    return s;
}

static __device__ __forceinline__ Dl h2_in(const H2Src& s, int id, const wreal* a,
                                           const wreal* ad, size_t at)
{
    wreal d = (wreal)0;
    if (s.mode == 1) d = ad[at];
    else if (s.mode == 0 && s.which == id && s.at == at) d = (wreal)1;
    return dl(a[at], d);
}

static __device__ __forceinline__ int h2_ix(const H2Src& s, int i)
{
    return s.bx ? max(0, min(i, s.nx - 1)) : df_per(i, s.nx);
}
static __device__ __forceinline__ int h2_iy(const H2Src& s, int j)
{
    return s.by ? max(0, min(j, s.ny - 1)) : df_per(j, s.ny);
}
static __device__ __forceinline__ int h2_iu(const H2Src& s, int i)
{
    return s.bx ? max(0, min(i, s.nx)) : df_per(i, s.nx);
}
static __device__ __forceinline__ int h2_jv(const H2Src& s, int j)
{
    return s.by ? max(0, min(j, s.ny)) : df_per(j, s.ny);
}

// ---- metrics (wrf_phi, wrf_exact_z_at_w, wrf_rdzw, wrf_rdz, wrf_zx, wrf_zy)

static __device__ Dl h2_zw(const H2Src& s, int kw, int j, int i)
{
    kw = max(0, min(kw, s.nz));
    size_t at = R3(kw, h2_iy(s, j), h2_ix(s, i), s.ny, s.nx);
    return (s.phb[at] + h2_in(s, H2_PHP, s.php, s.phpd, at)) / DF_G;
}

static __device__ Dl h2_rdzw_ev(const H2Src& s, int k, int j, int i)
{
    k = max(0, min(k, s.nz - 1));
    return (wreal)1 / (h2_zw(s, k + 1, j, i) - h2_zw(s, k, j, i));
}

static __device__ Dl h2_rdz_ev(const H2Src& s, int kw, int j, int i)
{
    if (kw <= 0) return (wreal)2 / (h2_zw(s, 1, j, i) - h2_zw(s, 0, j, i));
    if (kw >= s.nz) return dl((wreal)0, (wreal)0);   // WRF leaves the top row unwritten
    return (wreal)2 / (h2_zw(s, kw + 1, j, i) - h2_zw(s, kw - 1, j, i));
}

static __device__ Dl h2_zx_ev(const H2Src& s, int kw, int j, int iface)
{
    if (s.bx && iface <= 0) return dl((wreal)0, (wreal)0);
    if (s.bx && iface >= s.nx) iface = s.nx - 1;
    int jc = h2_iy(s, j), ip = h2_ix(s, iface), im = h2_ix(s, iface - 1);
    kw = max(0, min(kw, s.nz));
    size_t a = R3(kw, jc, ip, s.ny, s.nx), b = R3(kw, jc, im, s.ny, s.nx);
    wreal base = wdiv(s.rdx * (s.phb[a] - s.phb[b]), DF_G);
    Dl pert = (s.rdx * (h2_in(s, H2_PHP, s.php, s.phpd, a)
                        - h2_in(s, H2_PHP, s.php, s.phpd, b))) / DF_G;
    return base + pert;
}

static __device__ Dl h2_zy_ev(const H2Src& s, int kw, int jface, int i)
{
    if (s.by && jface <= 0) return dl((wreal)0, (wreal)0);
    if (s.by && jface >= s.ny) jface = s.ny - 1;
    int ic = h2_ix(s, i), jp = h2_iy(s, jface), jm = h2_iy(s, jface - 1);
    kw = max(0, min(kw, s.nz));
    size_t a = R3(kw, jp, ic, s.ny, s.nx), b = R3(kw, jm, ic, s.ny, s.nx);
    wreal base = wdiv(s.rdy * (s.phb[a] - s.phb[b]), DF_G);
    Dl pert = (s.rdy * (h2_in(s, H2_PHP, s.php, s.phpd, a)
                        - h2_in(s, H2_PHP, s.php, s.phpd, b))) / DF_G;
    return base + pert;
}

// The stored metrics, read as the primal's functions are called.
static __device__ __forceinline__ Dl h2_zx(const H2Src& s, int kw, int j, int iface)
{
    int f = s.bx ? max(0, min(iface, s.nx)) : df_per(iface, s.nx);
    return h2_in(s, H2_ZX, s.zx, s.zxd,
                 R3(max(0, min(kw, s.nz)), h2_iy(s, j), f, s.ny, s.nx + 1));
}
static __device__ __forceinline__ Dl h2_zy(const H2Src& s, int kw, int jface, int i)
{
    int f = s.by ? max(0, min(jface, s.ny)) : df_per(jface, s.ny);
    return h2_in(s, H2_ZY, s.zy, s.zyd,
                 R3(max(0, min(kw, s.nz)), f, h2_ix(s, i), s.ny + 1, s.nx));
}
static __device__ __forceinline__ Dl h2_rdzw(const H2Src& s, int k, int j, int i)
{
    return h2_in(s, H2_RZ, s.rz, s.rzd,
                 R3(max(0, min(k, s.nz - 1)), h2_iy(s, j), h2_ix(s, i), s.ny, s.nx));
}
static __device__ __forceinline__ Dl h2_rdz(const H2Src& s, int kw, int j, int i)
{
    return h2_in(s, H2_RZ, s.rz, s.rzd,
                 R3(max(0, min(kw, s.nz)), h2_iy(s, j), h2_ix(s, i), s.ny, s.nx));
}

// ---- density and coefficient (wrf_rho, wrf_k)

static __device__ Dl h2_rho_ev(const H2Src& s, size_t at)
{
    Dl inv = (wreal)1 / h2_in(s, H2_ALT, s.alt, s.altd, at);
    if (s.moist) return inv * ((wreal)1 + h2_in(s, H2_QV, s.qv, s.qvd, at));
    return inv;
}
static __device__ __forceinline__ Dl h2_rho(const H2Src& s, int k, int j, int i)
{
    return h2_in(s, H2_RHO, s.rho, s.rhod, R3(k, h2_iy(s, j), h2_ix(s, i), s.ny, s.nx));
}
static __device__ __forceinline__ Dl h2_k(const H2Src& s, int k, int j, int i)
{
    return h2_in(s, H2_K, s.km, s.kmd, R3(k, h2_iy(s, j), h2_ix(s, i), s.ny, s.nx));
}

// ---- u and v stresses (wrf_tau11, wrf_tau22, wrf_tau12)

static __device__ Dl h2_ta_ev(const H2Src& s, int k, int j, int i)
{
    return ((-h2_rho(s, k, j, i)) * h2_k(s, k, j, i))
           * h2_in(s, H2_DA, s.da, s.dad, R3(k, j, i, s.ny, s.nx));
}

static __device__ Dl h2_t12_ev(const H2Src& s, int k, int j, int i)
{
    Dl rhoavg = (wreal)0.25 * (((h2_rho(s, k, j, i) + h2_rho(s, k, j, i - 1))
                                + h2_rho(s, k, j - 1, i - 1)) + h2_rho(s, k, j - 1, i));
    Dl kavg = (wreal)0.25 * (((h2_k(s, k, j, i) + h2_k(s, k, j, i - 1))
                              + h2_k(s, k, j - 1, i - 1)) + h2_k(s, k, j - 1, i));
    return ((-rhoavg) * kavg)
           * h2_in(s, H2_D12, s.d12, s.d12d, R3(k, h2_iy(s, j), h2_ix(s, i), s.ny, s.nx));
}

// ---- w stresses (wrf_what, wrf_w_xavg/yavg, wrf_defor13/23, wrf_tau13/23)

static __device__ __forceinline__ Dl h2_what(const H2Src& s, int kw, int j, int i)
{
    int jj = h2_iy(s, j), ii = h2_ix(s, i);
    return h2_in(s, H2_W, s.w, s.wd, R3(kw, jj, ii, s.ny, s.nx)) / s.msft[(size_t)jj * s.nx + ii];
}

static __device__ Dl h2_defor13(const H2Src& s, int kw, int j, int i)
{
    if (kw <= 0 || kw >= s.nz) return dl((wreal)0, (wreal)0);
    if (s.bx) i = i <= 0 ? (s.nx > 1 ? 1 : 0) : (i >= s.nx ? s.nx - 1 : i);
    if (s.by) j = j <= 0 ? (s.ny > 1 ? 1 : 0) : (j >= s.ny ? s.ny - 1 : j);
    int jj = h2_iy(s, j), iu = h2_iu(s, i);
    wreal msf = s.msfu[(size_t)jj * (s.nx + 1) + iu];
    Dl rz = (wreal)0.5 * (h2_rdz(s, kw, j, i) + h2_rdz(s, kw, j, i - 1));
    Dl upper = (wreal)0.25 * (((h2_what(s, kw, j, i) + h2_what(s, kw + 1, j, i))
                               + h2_what(s, kw, j, i - 1)) + h2_what(s, kw + 1, j, i - 1));
    Dl lower = (wreal)0.25 * (((h2_what(s, kw - 1, j, i) + h2_what(s, kw, j, i))
                               + h2_what(s, kw - 1, j, i - 1)) + h2_what(s, kw, j, i - 1));
    Dl slope = ((upper - lower) * h2_zx(s, kw, j, i)) * rz;
    Dl dwdx = (msf * msf) * ((s.rdx * (h2_what(s, kw, j, i) - h2_what(s, kw, j, i - 1))) - slope);
    Dl dudz = (h2_in(s, H2_U, s.u, s.ud, R3(kw, jj, iu, s.ny, s.nx + 1))
               - h2_in(s, H2_U, s.u, s.ud, R3(kw - 1, jj, iu, s.ny, s.nx + 1))) * rz;
    return dwdx + dudz;
}

static __device__ Dl h2_defor23(const H2Src& s, int kw, int j, int i)
{
    if (kw <= 0 || kw >= s.nz) return dl((wreal)0, (wreal)0);
    if (s.bx) i = i <= 0 ? (s.nx > 1 ? 1 : 0) : (i >= s.nx ? s.nx - 1 : i);
    if (s.by) j = j <= 0 ? (s.ny > 1 ? 1 : 0) : (j >= s.ny ? s.ny - 1 : j);
    int jv = h2_jv(s, j), ii = h2_ix(s, i);
    wreal msf = s.msfv[(size_t)jv * s.nx + ii];
    Dl rz = (wreal)0.5 * (h2_rdz(s, kw, j, i) + h2_rdz(s, kw, j - 1, i));
    Dl upper = (wreal)0.25 * (((h2_what(s, kw, j, i) + h2_what(s, kw + 1, j, i))
                               + h2_what(s, kw, j - 1, i)) + h2_what(s, kw + 1, j - 1, i));
    Dl lower = (wreal)0.25 * (((h2_what(s, kw - 1, j, i) + h2_what(s, kw, j, i))
                               + h2_what(s, kw - 1, j - 1, i)) + h2_what(s, kw, j - 1, i));
    Dl slope = ((upper - lower) * h2_zy(s, kw, j, i)) * rz;
    Dl dwdy = (msf * msf) * ((s.rdy * (h2_what(s, kw, j, i) - h2_what(s, kw, j - 1, i))) - slope);
    Dl dvdz = (h2_in(s, H2_V, s.v, s.vd, R3(kw, jv, ii, s.ny + 1, s.nx))
               - h2_in(s, H2_V, s.v, s.vd, R3(kw - 1, jv, ii, s.ny + 1, s.nx))) * rz;
    return dwdy + dvdz;
}

static __device__ Dl h2_t13_ev(const H2Src& s, int kw, int j, int i)
{
    if (kw <= 0 || kw >= s.nz) return dl((wreal)0, (wreal)0);
    Dl rhoavg = (wreal)0.5 * ((s.fnm[kw] * (h2_rho(s, kw, j, i - 1) + h2_rho(s, kw, j, i)))
                              + (s.fnp[kw] * (h2_rho(s, kw - 1, j, i - 1)
                                              + h2_rho(s, kw - 1, j, i))));
    Dl kavg = (wreal)0.5 * ((s.fnm[kw] * (h2_k(s, kw, j, i - 1) + h2_k(s, kw, j, i)))
                            + (s.fnp[kw] * (h2_k(s, kw - 1, j, i - 1) + h2_k(s, kw - 1, j, i))));
    return ((-rhoavg) * kavg) * h2_defor13(s, kw, j, i);
}

static __device__ Dl h2_t23_ev(const H2Src& s, int kw, int j, int i)
{
    if (kw <= 0 || kw >= s.nz) return dl((wreal)0, (wreal)0);
    Dl rhoavg = (wreal)0.5 * ((s.fnm[kw] * (h2_rho(s, kw, j - 1, i) + h2_rho(s, kw, j, i)))
                              + (s.fnp[kw] * (h2_rho(s, kw - 1, j - 1, i)
                                              + h2_rho(s, kw - 1, j, i))));
    Dl kavg = (wreal)0.5 * ((s.fnm[kw] * (h2_k(s, kw, j - 1, i) + h2_k(s, kw, j, i)))
                            + (s.fnp[kw] * (h2_k(s, kw - 1, j - 1, i) + h2_k(s, kw - 1, j, i))));
    return ((-rhoavg) * kavg) * h2_defor23(s, kw, j, i);
}

// ---- scalar fluxes (wrf_scalar, wrf_scalar_w_xface/yface, wrf_h1/h2)

static __device__ __forceinline__ Dl h2_sc(const H2Src& s, int k, int j, int i)
{
    return h2_in(s, H2_F, s.f, s.fd, R3(k, h2_iy(s, j), h2_ix(s, i), s.ny, s.nx));
}

// The field at w level kw on the face between (ja, ia) and (jb, ib)
// (= (j, i) and (j, i - 1) or (j - 1, i)).
static __device__ Dl h2_face(const H2Src& s, int kw, int ja, int ia, int jb, int ib)
{
    if (kw <= 0)
        return (wreal)0.5 * (((((s.cf1 * h2_sc(s, 0, ja, ia) + s.cf2 * h2_sc(s, 1, ja, ia))
                                + s.cf3 * h2_sc(s, 2, ja, ia)) + s.cf1 * h2_sc(s, 0, jb, ib))
                              + s.cf2 * h2_sc(s, 1, jb, ib)) + s.cf3 * h2_sc(s, 2, jb, ib));
    if (kw >= s.nz) {
        wreal hw = s.dnw[s.nz - 1], dd = s.dn[s.nz - 1];
        Dl a = h2_sc(s, s.nz - 1, ja, ia), b = h2_sc(s, s.nz - 2, ja, ia);
        Dl c = h2_sc(s, s.nz - 1, jb, ib), d = h2_sc(s, s.nz - 2, jb, ib);
        return (wreal)0.5 * (((a + (((a - b) * (wreal)0.5) * hw) / dd) + c)
                             + (((c - d) * (wreal)0.5) * hw) / dd);
    }
    Dl pc = h2_sc(s, kw, jb, ib) + h2_sc(s, kw, ja, ia);
    Dl pb = h2_sc(s, kw - 1, jb, ib) + h2_sc(s, kw - 1, ja, ia);
    return (wreal)0.5 * ((s.fnm[kw] * pc) + (s.fnp[kw] * pb));
}

static __device__ Dl h2_h1_ev(const H2Src& s, int k, int j, int i)
{
    Dl kavg = (wreal)0.5 * (h2_k(s, k, j, i - 1) + h2_k(s, k, j, i));
    Dl tmpzx = (wreal)0.5 * (h2_zx(s, k, j, i) + h2_zx(s, k + 1, j, i));
    Dl rdzu = (wreal)2 / (((wreal)1 / h2_rdzw(s, k, j, i)) + ((wreal)1 / h2_rdzw(s, k, j, i - 1)));
    wreal map = s.msfu[(size_t)h2_iy(s, j) * (s.nx + 1) + h2_iu(s, i)];
    Dl grad = (s.rdx * (h2_sc(s, k, j, i) - h2_sc(s, k, j, i - 1)))
              - ((tmpzx * (h2_face(s, k + 1, j, i, j, i - 1) - h2_face(s, k, j, i, j, i - 1)))
                 * rdzu);
    Dl xkxavg = (kavg * (wreal)0.5) * (h2_rho(s, k, j, i - 1) + h2_rho(s, k, j, i));
    return ((-map) * xkxavg) * grad;
}

static __device__ Dl h2_h2_ev(const H2Src& s, int k, int j, int i)
{
    Dl kavg = (wreal)0.5 * (h2_k(s, k, j - 1, i) + h2_k(s, k, j, i));
    Dl tmpzy = (wreal)0.5 * (h2_zy(s, k, j, i) + h2_zy(s, k + 1, j, i));
    Dl rdzv = (wreal)2 / (((wreal)1 / h2_rdzw(s, k, j, i)) + ((wreal)1 / h2_rdzw(s, k, j - 1, i)));
    wreal map = s.msfv[(size_t)h2_jv(s, j) * s.nx + h2_ix(s, i)];
    Dl grad = (s.rdy * (h2_sc(s, k, j, i) - h2_sc(s, k, j - 1, i)))
              - ((tmpzy * (h2_face(s, k + 1, j, i, j - 1, i) - h2_face(s, k, j, i, j - 1, i)))
                 * rdzv);
    Dl xkxavg = (kavg * (wreal)0.5) * (h2_rho(s, k, j - 1, i) + h2_rho(s, k, j, i));
    return ((-map) * xkxavg) * grad;
}

// ---- the rows (wrf_smag_hd_u, wrf_smag_hd_v, wrf_smag_hd_w, wrf_smag_hd_s)

// the u and v rows' stresses: s1 = tau11 or tau22 (mass), s2 = tau12
// (corners, read at the corner itself)
static __device__ __forceinline__ Dl h2_ta(const H2Src& s, int k, int j, int i)
{
    return h2_in(s, H2_S1, s.s1, s.s1d, R3(k, h2_iy(s, j), h2_ix(s, i), s.ny, s.nx));
}
static __device__ __forceinline__ Dl h2_t12(const H2Src& s, int k, int j, int i)
{
    return h2_in(s, H2_S2, s.s2, s.s2d, R3(k, j, i, s.ny + 1, s.nx + 1));
}

// fnm/fnp interpolation of a stress pair to w level kw (0 off the interior)
static __device__ __forceinline__ Dl h2_wavg(const H2Src& s, int kw, Dl upper, Dl lower)
{
    return (wreal)0.5 * ((s.fnm[kw] * upper) + (s.fnp[kw] * lower));
}

static __device__ Dl h2_row_u(const H2Src& s, int k, int j, int i)
{
    int ic = h2_ix(s, i), im = h2_ix(s, i - 1), iu = h2_iu(s, i), jj = h2_iy(s, j);
    wreal msf = s.msfu[(size_t)jj * (s.nx + 1) + iu];
    Dl tmpdz = (wreal)0.5 * (((wreal)1 / h2_rdzw(s, k, j, ic)) + ((wreal)1 / h2_rdzw(s, k, j, im)));
    Dl zxu = (wreal)0.5 * (h2_zx(s, k, j, i) + h2_zx(s, k + 1, j, i));
    Dl zyu = (wreal)0.125 * (((((((h2_zy(s, k, j, im) + h2_zy(s, k, j, ic))
                                  + h2_zy(s, k, j + 1, im)) + h2_zy(s, k, j + 1, ic))
                                + h2_zy(s, k + 1, j, im)) + h2_zy(s, k + 1, j, ic))
                              + h2_zy(s, k + 1, j + 1, im)) + h2_zy(s, k + 1, j + 1, ic));
    Dl divh = ((msf * s.rdx) * (h2_ta(s, k, j, ic) - h2_ta(s, k, j, im)))
              + ((msf * s.rdy) * (h2_t12(s, k, j + 1, i) - h2_t12(s, k, j, i)));
    Dl a11[2], a12[2];
    for (int q = 0; q < 2; ++q) {
        int kw = k + 1 - q;
        if (kw <= 0 || kw >= s.nz) {
            a11[q] = dl((wreal)0, (wreal)0);
            a12[q] = dl((wreal)0, (wreal)0);
            continue;
        }
        a11[q] = h2_wavg(s, kw, h2_ta(s, kw, j, i - 1) + h2_ta(s, kw, j, i),
                         h2_ta(s, kw - 1, j, i - 1) + h2_ta(s, kw - 1, j, i));
        a12[q] = h2_wavg(s, kw, h2_t12(s, kw, j + 1, i) + h2_t12(s, kw, j, i),
                         h2_t12(s, kw - 1, j + 1, i) + h2_t12(s, kw - 1, j, i));
    }
    Dl first = ((msf * zxu) * (a11[0] - a11[1])) / tmpdz;
    Dl second = ((msf * zyu) * (a12[0] - a12[1])) / tmpdz;
    return ((DF_G * tmpdz) / s.dnw[k]) * ((divh - first) - second);
}

static __device__ Dl h2_row_v(const H2Src& s, int k, int j, int i)
{
    int jc = h2_iy(s, j), jm = h2_iy(s, j - 1), jv = h2_jv(s, j), ii = h2_ix(s, i);
    wreal msf = s.msfv[(size_t)jv * s.nx + ii];
    Dl tmpdz = (wreal)0.5 * (((wreal)1 / h2_rdzw(s, k, jc, i)) + ((wreal)1 / h2_rdzw(s, k, jm, i)));
    Dl zxv = (wreal)0.125 * (((((((h2_zx(s, k, jc, i) + h2_zx(s, k, jc, i + 1))
                                  + h2_zx(s, k, jm, i)) + h2_zx(s, k, jm, i + 1))
                                + h2_zx(s, k + 1, jc, i)) + h2_zx(s, k + 1, jc, i + 1))
                              + h2_zx(s, k + 1, jm, i)) + h2_zx(s, k + 1, jm, i + 1));
    Dl zyv = (wreal)0.5 * (h2_zy(s, k, j, i) + h2_zy(s, k + 1, j, i));
    Dl divh = ((msf * s.rdy) * (h2_ta(s, k, jc, i) - h2_ta(s, k, jm, i)))
              + ((msf * s.rdx) * (h2_t12(s, k, j, i + 1) - h2_t12(s, k, j, i)));
    Dl a12[2], a22[2];
    for (int q = 0; q < 2; ++q) {
        int kw = k + 1 - q;
        if (kw <= 0 || kw >= s.nz) {
            a12[q] = dl((wreal)0, (wreal)0);
            a22[q] = dl((wreal)0, (wreal)0);
            continue;
        }
        a12[q] = h2_wavg(s, kw, h2_t12(s, kw, j, i + 1) + h2_t12(s, kw, j, i),
                         h2_t12(s, kw - 1, j, i + 1) + h2_t12(s, kw - 1, j, i));
        a22[q] = h2_wavg(s, kw, h2_ta(s, kw, j - 1, i) + h2_ta(s, kw, j, i),
                         h2_ta(s, kw - 1, j - 1, i) + h2_ta(s, kw - 1, j, i));
    }
    Dl first = ((msf * zxv) * (a12[0] - a12[1])) / tmpdz;
    Dl second = ((msf * zyv) * (a22[0] - a22[1])) / tmpdz;
    return ((DF_G * tmpdz) / s.dnw[k]) * ((divh - first) - second);
}

// the w row's stresses, stored on (nz + 1) w levels at the faces
static __device__ __forceinline__ Dl h2_t13(const H2Src& s, int kw, int j, int i)
{
    return h2_in(s, H2_S1, s.s1, s.s1d, R3(kw, j, i, s.ny, s.nx + 1));
}
static __device__ __forceinline__ Dl h2_t23(const H2Src& s, int kw, int j, int i)
{
    return h2_in(s, H2_S2, s.s2, s.s2d, R3(kw, j, i, s.ny + 1, s.nx));
}

static __device__ Dl h2_row_w(const H2Src& s, int kw, int j, int i)
{
    if (kw <= 0 || kw >= s.nz) return dl((wreal)0, (wreal)0);
    wreal map = s.msft[(size_t)j * s.nx + i];
    Dl zxw = (wreal)0.5 * (h2_zx(s, kw, j, i) + h2_zx(s, kw, j, i + 1));
    Dl zyw = (wreal)0.5 * (h2_zy(s, kw, j, i) + h2_zy(s, kw, j + 1, i));
    Dl rz = h2_rdz(s, kw, j, i);
    Dl divh = ((map * s.rdx) * (h2_t13(s, kw, j, i + 1) - h2_t13(s, kw, j, i)))
              + ((map * s.rdy) * (h2_t23(s, kw, j + 1, i) - h2_t23(s, kw, j, i)));
    Dl m13[2], m23[2];
    for (int q = 0; q < 2; ++q) {
        int k = kw - q;
        m13[q] = (wreal)0.25 * (((h2_t13(s, k + 1, j, i + 1) + h2_t13(s, k + 1, j, i))
                                 + h2_t13(s, k, j, i + 1)) + h2_t13(s, k, j, i));
        m23[q] = (wreal)0.25 * (((h2_t23(s, k + 1, j + 1, i) + h2_t23(s, k + 1, j, i))
                                 + h2_t23(s, k, j + 1, i)) + h2_t23(s, k, j, i));
    }
    Dl divz = (map * rz) * ((zxw * (m13[0] - m13[1])) + (zyw * (m23[0] - m23[1])));
    return (DF_G / (s.dn[kw] * rz)) * (divh - divz);
}

// the scalar fluxes, read as wrf_fx / wrf_fy
static __device__ __forceinline__ Dl h2_fx(const H2Src& s, int k, int j, int i)
{
    return h2_in(s, H2_S1, s.s1, s.s1d, R3(k, h2_iy(s, j), h2_iu(s, i), s.ny, s.nx + 1));
}
static __device__ __forceinline__ Dl h2_fy(const H2Src& s, int k, int j, int i)
{
    return h2_in(s, H2_S2, s.s2, s.s2d, R3(k, h2_jv(s, j), h2_ix(s, i), s.ny + 1, s.nx));
}

static __device__ Dl h2_row_s(const H2Src& s, int k, int j, int i)
{
    wreal map = s.msft[(size_t)j * s.nx + i];
    Dl zxm = (wreal)0.25 * (((h2_zx(s, k, j, i) + h2_zx(s, k, j, i + 1)) + h2_zx(s, k + 1, j, i))
                            + h2_zx(s, k + 1, j, i + 1));
    Dl zym = (wreal)0.25 * (((h2_zy(s, k, j, i) + h2_zy(s, k, j + 1, i)) + h2_zy(s, k + 1, j, i))
                            + h2_zy(s, k + 1, j + 1, i));
    Dl rz = h2_rdzw(s, k, j, i);
    Dl divh = ((map * s.rdx) * (h2_fx(s, k, j, i + 1) - h2_fx(s, k, j, i)))
              + ((map * s.rdy) * (h2_fy(s, k, j + 1, i) - h2_fy(s, k, j, i)));
    Dl m1[2], m2[2];
    for (int q = 0; q < 2; ++q) {
        int kw = k + 1 - q;
        if (kw <= 0 || kw >= s.nz) {
            m1[q] = dl((wreal)0, (wreal)0);
            m2[q] = dl((wreal)0, (wreal)0);
            continue;
        }
        m1[q] = h2_wavg(s, kw, h2_fx(s, kw, j, i + 1) + h2_fx(s, kw, j, i),
                        h2_fx(s, kw - 1, j, i + 1) + h2_fx(s, kw - 1, j, i));
        m2[q] = h2_wavg(s, kw, h2_fy(s, kw, j + 1, i) + h2_fy(s, kw, j, i),
                        h2_fy(s, kw - 1, j + 1, i) + h2_fy(s, kw - 1, j, i));
    }
    Dl first = ((map * zxm) * (m1[0] - m1[1])) * rz;
    Dl second = ((map * zym) * (m2[0] - m2[1])) * rz;
    return (DF_G / (s.dnw[k] * rz)) * ((divh - first) - second);
}

// ---- dispatch

static __device__ Dl h2_eval(const H2Src& s, int o, int k, int j, int i)
{
    switch (o) {
    case H2O_ZX: return h2_zx_ev(s, k, j, i);
    case H2O_ZY: return h2_zy_ev(s, k, j, i);
    case H2O_RDZW: return h2_rdzw_ev(s, k, j, i);
    case H2O_RDZ: return h2_rdz_ev(s, k, j, i);
    case H2O_RHO: return h2_rho_ev(s, R3(k, j, i, s.ny, s.nx));
    case H2O_TA: return h2_ta_ev(s, k, j, i);
    case H2O_T12: return h2_t12_ev(s, k, j, i);
    case H2O_T13: return h2_t13_ev(s, k, j, i);
    case H2O_T23: return h2_t23_ev(s, k, j, i);
    case H2O_H1: return h2_h1_ev(s, k, j, i);
    case H2O_H2: return h2_h2_ev(s, k, j, i);
    default: break;
    }
    if (s.row == 1) return h2_row_u(s, k, j, i);
    if (s.row == 2) return h2_row_v(s, k, j, i);
    if (s.row == 3) return h2_row_w(s, k, j, i);
    return h2_row_s(s, k, j, i);
}

// an output's levels and horizontal staggers
static __device__ void h2_odims(const H2Src& s, int o, int* nl, int* sy, int* sx)
{
    int L = s.nz, y = 0, x = 0;
    switch (o) {
    case H2O_ZX: L = s.nz + 1; x = 1; break;
    case H2O_ZY: L = s.nz + 1; y = 1; break;
    case H2O_RDZ: L = s.nz + 1; break;
    case H2O_T12: y = 1; x = 1; break;
    case H2O_T13: L = s.nz + 1; x = 1; break;
    case H2O_T23: L = s.nz + 1; y = 1; break;
    case H2O_H1: x = 1; break;
    case H2O_H2: y = 1; break;
    case H2O_TEND:
        if (s.row == 1) x = 1;
        else if (s.row == 2) y = 1;
        else if (s.row == 3) L = s.nz + 1;
        break;
    default: break;
    }
    *nl = L; *sy = y; *sx = x;
}

// the stage's outputs, in the order of its kernels' output arguments
static __device__ int h2_outputs(const H2Src& s, int stage, int* outs)
{
    switch (stage) {
    case H2S_MET: outs[0] = H2O_ZX; outs[1] = H2O_ZY; outs[2] = H2O_RDZW; outs[3] = H2O_RDZ;
        return 4;
    case H2S_RHO: outs[0] = H2O_RHO; return 1;
    case H2S_TUV: outs[0] = H2O_TA; outs[1] = H2O_T12; return 2;
    case H2S_TW: outs[0] = H2O_T13; outs[1] = H2O_T23; return 2;
    case H2S_HS: outs[0] = H2O_H1; outs[1] = H2O_H2; return 2;
    default: outs[0] = H2O_TEND; return 1;
    }
}

// the stage's active inputs, in the order of its AD kernel's adjoints
static __device__ int h2_inputs(const H2Src& s, int stage, int* ids)
{
    switch (stage) {
    case H2S_MET: ids[0] = H2_PHP; return 1;
    case H2S_RHO: ids[0] = H2_ALT; ids[1] = H2_QV; return s.moist ? 2 : 1;
    case H2S_TUV: ids[0] = H2_RHO; ids[1] = H2_K; ids[2] = H2_DA; ids[3] = H2_D12; return 4;
    case H2S_TW:
        ids[0] = H2_U; ids[1] = H2_V; ids[2] = H2_W; ids[3] = H2_ZX; ids[4] = H2_ZY;
        ids[5] = H2_RZ; ids[6] = H2_RHO; ids[7] = H2_K;
        return 8;
    case H2S_HS:
        ids[0] = H2_F; ids[1] = H2_K; ids[2] = H2_RHO; ids[3] = H2_ZX; ids[4] = H2_ZY;
        ids[5] = H2_RZ;
        return 6;
    default:
        ids[0] = H2_S1; ids[1] = H2_S2; ids[2] = H2_ZX; ids[3] = H2_ZY; ids[4] = H2_RZ;
        return 5;
    }
}

// an input's levels and horizontal staggers within a stage
static __device__ void h2_idims(const H2Src& s, int stage, int id, int* nl, int* sy, int* sx)
{
    int L = s.nz, y = 0, x = 0;
    switch (id) {
    case H2_PHP: case H2_W: L = s.nz + 1; break;
    case H2_U: x = 1; break;
    case H2_V: y = 1; break;
    case H2_ZX: L = s.nz + 1; x = 1; break;
    case H2_ZY: L = s.nz + 1; y = 1; break;
    case H2_RZ:   // rdz on w levels for the w stresses and row, else rdzw
        if (stage == H2S_TW || (stage == H2S_ROW && s.row == 3)) L = s.nz + 1;
        break;
    case H2_S1: case H2_S2:
        if (s.row == 3) { L = s.nz + 1; if (id == H2_S1) x = 1; else y = 1; }
        else if (s.row == 0) { if (id == H2_S1) x = 1; else y = 1; }
        else if (id == H2_S2) { y = 1; x = 1; }
        break;
    default: break;
    }
    *nl = L; *sy = y; *sx = x;
}

// The output offsets (levels, rows, columns; lo..hi) whose stencil can read
// an input word: per (input, output) of the stage, from the primal's reads
// with its clamps and bounded-face copies.  Returns the number of outputs
// the input feeds; slot[] indexes h2_outputs.
static __device__ int h2_pairs(const H2Src& s, int stage, int id, int* slot, int* win)
{
    int n = 0;
#define H2_PAIR(sl, a, b, c, d, e, f) \
    { slot[n] = sl; win[6 * n] = a; win[6 * n + 1] = b; win[6 * n + 2] = c; \
      win[6 * n + 3] = d; win[6 * n + 4] = e; win[6 * n + 5] = f; ++n; }
    if (stage == H2S_MET) {
        H2_PAIR(0, 0, 0, 0, 0, 0, 2);      // zx: faces i, i+1 and the copied end face
        H2_PAIR(1, 0, 0, 0, 2, 0, 0);
        H2_PAIR(2, -1, 0, 0, 0, 0, 0);     // rdzw: levels kw-1, kw
        H2_PAIR(3, -1, 1, 0, 0, 0, 0);     // rdz: kw-1, kw+1 and the bottom row
    } else if (stage == H2S_RHO) {
        H2_PAIR(0, 0, 0, 0, 0, 0, 0);
    } else if (stage == H2S_TUV) {
        if (id == H2_RHO || id == H2_K) {
            H2_PAIR(0, 0, 0, 0, 0, 0, 0);
            H2_PAIR(1, 0, 0, 0, 1, 0, 1);
        } else if (id == H2_DA) {
            H2_PAIR(0, 0, 0, 0, 0, 0, 0);
        } else {
            H2_PAIR(1, 0, 0, 0, 1, 0, 1);
        }
    } else if (stage == H2S_TW) {
        switch (id) {
        case H2_RHO: case H2_K:
            H2_PAIR(0, 0, 1, 0, 0, 0, 1);
            H2_PAIR(1, 0, 1, 0, 1, 0, 0);
            break;
        case H2_W:
            H2_PAIR(0, -1, 1, -1, 0, -1, 2);
            H2_PAIR(1, -1, 1, -1, 2, -1, 0);
            break;
        case H2_U: H2_PAIR(0, 0, 1, -1, 0, -1, 1); break;
        case H2_V: H2_PAIR(1, 0, 1, -1, 1, -1, 0); break;
        case H2_ZX: H2_PAIR(0, 0, 0, -1, 0, -1, 1); break;
        case H2_ZY: H2_PAIR(1, 0, 0, -1, 1, -1, 0); break;
        default:   // rdz
            H2_PAIR(0, 0, 0, -1, 0, -1, 2);
            H2_PAIR(1, 0, 0, -1, 2, -1, 0);
            break;
        }
    } else if (stage == H2S_HS) {
        switch (id) {
        case H2_F:
            H2_PAIR(0, -2, 1, 0, 0, 0, 1);
            H2_PAIR(1, -2, 1, 0, 1, 0, 0);
            break;
        case H2_ZX: H2_PAIR(0, -1, 0, 0, 0, 0, 0); break;
        case H2_ZY: H2_PAIR(1, -1, 0, 0, 0, 0, 0); break;
        default:   // K, rho, rdzw
            H2_PAIR(0, 0, 0, 0, 0, 0, 1);
            H2_PAIR(1, 0, 0, 0, 1, 0, 0);
            break;
        }
    } else if (s.row == 1) {
        switch (id) {
        case H2_S1: H2_PAIR(0, -1, 1, 0, 0, 0, 1); break;
        case H2_S2: H2_PAIR(0, -1, 1, -1, 0, 0, 0); break;
        case H2_ZX: H2_PAIR(0, -1, 0, 0, 0, 0, 0); break;
        case H2_ZY: H2_PAIR(0, -1, 0, -1, 0, 0, 1); break;
        default: H2_PAIR(0, 0, 0, 0, 0, 0, 1); break;
        }
    } else if (s.row == 2) {
        switch (id) {
        case H2_S1: H2_PAIR(0, -1, 1, 0, 1, 0, 0); break;
        case H2_S2: H2_PAIR(0, -1, 1, 0, 0, -1, 0); break;
        case H2_ZX: H2_PAIR(0, -1, 0, 0, 1, -1, 0); break;
        case H2_ZY: H2_PAIR(0, -1, 0, 0, 0, 0, 0); break;
        default: H2_PAIR(0, 0, 0, 0, 1, 0, 0); break;
        }
    } else if (s.row == 3) {
        switch (id) {
        case H2_S1: H2_PAIR(0, -1, 1, 0, 0, -1, 0); break;
        case H2_S2: H2_PAIR(0, -1, 1, -1, 0, 0, 0); break;
        case H2_ZX: H2_PAIR(0, 0, 0, 0, 0, -1, 0); break;
        case H2_ZY: H2_PAIR(0, 0, 0, -1, 0, 0, 0); break;
        default: H2_PAIR(0, 0, 0, 0, 0, 0, 0); break;
        }
    } else {
        switch (id) {
        case H2_S1: H2_PAIR(0, -1, 1, 0, 0, -1, 0); break;
        case H2_S2: H2_PAIR(0, -1, 1, -1, 0, 0, 0); break;
        case H2_ZX: H2_PAIR(0, -1, 0, 0, 0, -1, 0); break;
        case H2_ZY: H2_PAIR(0, -1, 0, -1, 0, 0, 0); break;
        default: H2_PAIR(0, 0, 0, 0, 0, 0, 0); break;
        }
    }
#undef H2_PAIR
    return n;
}

// The distinct output coordinates c + lo .. c + hi on an axis of n mass
// points (n + stag outputs): bounded, those in range; periodic, the images
// mod n, and the staggered end point n with image 0.
static __device__ int h2_cand(int c, int lo, int hi, int n, int stag, int bounded, int* out)
{
    int m = 0;
    for (int off = lo; off <= hi; ++off) {
        int x = c + off;
        if (bounded) {
            if (x >= 0 && x < n + stag) out[m++] = x;
            continue;
        }
        x = df_per(x, n);
        bool seen = false;
        for (int q = 0; q < m; ++q) seen = seen || out[q] == x;
        if (!seen) out[m++] = x;
        if (stag && x == 0) {
            seen = false;
            for (int q = 0; q < m; ++q) seen = seen || out[q] == n;
            if (!seen) out[m++] = n;
        }
    }
    return m;
}

// sum_out a_out * d out / d (the seeded word at (k, j, i) of input id)
static __device__ wreal h2_gather(const H2Src& s, int stage, int id, int k, int j, int i,
                                  const wreal* a0, const wreal* a1, const wreal* a2,
                                  const wreal* a3)
{
    int outs[4], slot[4], win[24];
    h2_outputs(s, stage, outs);
    int np = h2_pairs(s, stage, id, slot, win);
    wreal acc = (wreal)0;
    for (int p = 0; p < np; ++p) {
        int o = outs[slot[p]];
        const wreal* a = slot[p] == 0 ? a0 : slot[p] == 1 ? a1 : slot[p] == 2 ? a2 : a3;
        int nl, sy, sx;
        h2_odims(s, o, &nl, &sy, &sx);
        int ks[8], js[8], is[8];
        int mk = h2_cand(k, win[6 * p], win[6 * p + 1], nl, 0, 1, ks);
        int mj = h2_cand(j, win[6 * p + 2], win[6 * p + 3], s.ny, sy, s.by, js);
        int mi = h2_cand(i, win[6 * p + 4], win[6 * p + 5], s.nx, sx, s.bx, is);
        for (int c = 0; c < mk; ++c)
            for (int b = 0; b < mj; ++b)
                for (int q = 0; q < mi; ++q) {
                    size_t at = R3(ks[c], js[b], is[q], s.ny + sy, s.nx + sx);
                    if (a[at] != (wreal)0)
                        acc = acc + a[at] * h2_eval(s, o, ks[c], js[b], is[q]).d;
                }
    }
    return acc;
}

// One AD thread: decode word t of the stage's concatenated inputs, seed it
// and gather.  *slot_out is its position in h2_inputs, *at_out its word.
static __device__ bool h2_ad_word(H2Src& s, int stage, size_t t, const wreal* a0,
                                  const wreal* a1, const wreal* a2, const wreal* a3,
                                  int* slot_out, size_t* at_out, wreal* acc)
{
    int ids[8];
    int ni = h2_inputs(s, stage, ids);
    for (int q = 0; q < ni; ++q) {
        int nl, sy, sx;
        h2_idims(s, stage, ids[q], &nl, &sy, &sx);
        size_t ny = (size_t)(s.ny + sy), nx = (size_t)(s.nx + sx), size = (size_t)nl * ny * nx;
        if (t >= size) {
            t -= size;
            continue;
        }
        int i = (int)(t % nx), j = (int)((t / nx) % ny), k = (int)(t / (nx * ny));
        s.which = ids[q];
        s.at = t;
        *slot_out = q;
        *at_out = t;
        *acc = h2_gather(s, stage, ids[q], k, j, i, a0, a1, a2, a3);
        return true;
    }
    return false;
}

#define H2_GEOM wreal rdx, wreal rdy, wreal cf1, wreal cf2, wreal cf3, int bx, int by, int nz, \
    int ny, int nx, int row, int moist
#define H2_GARGS rdx, rdy, cf1, cf2, cf3, bx, by, nz, ny, nx, row, moist

// One forward thread: output word t of the stage's concatenated outputs.
static __device__ bool h2_fwd_word(const H2Src& s, int stage, size_t t, int* slot_out,
                                   size_t* at_out, Dl* r)
{
    int outs[4];
    int no = h2_outputs(s, stage, outs);
    for (int q = 0; q < no; ++q) {
        int nl, sy, sx;
        h2_odims(s, outs[q], &nl, &sy, &sx);
        size_t ny = (size_t)(s.ny + sy), nx = (size_t)(s.nx + sx), size = (size_t)nl * ny * nx;
        if (t >= size) {
            t -= size;
            continue;
        }
        int i = (int)(t % nx), j = (int)((t / nx) % ny), k = (int)(t / (nx * ny));
        *slot_out = q;
        *at_out = t;
        *r = h2_eval(s, outs[q], k, j, i);
        return true;
    }
    return false;
}

// ---- kernels.  ``tl`` 1: values and tangents; 0: values only (the AD's
// base intermediates).

extern "C" __global__ void h2_met_tl(const wreal* __restrict__ php,
                                     const wreal* __restrict__ phpd,
                                     const wreal* __restrict__ phb, wreal* __restrict__ zx,
                                     wreal* __restrict__ zxd, wreal* __restrict__ zy,
                                     wreal* __restrict__ zyd, wreal* __restrict__ rdzw,
                                     wreal* __restrict__ rdzwd, wreal* __restrict__ rdz,
                                     wreal* __restrict__ rdzd, int tl, H2_GEOM)
{
    H2Src s = h2_src(H2_GARGS, tl ? 1 : 2);
    s.php = php; s.phpd = phpd; s.phb = phb;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int q;
    size_t at;
    Dl r;
    if (!h2_fwd_word(s, H2S_MET, t, &q, &at, &r)) return;
    wreal* v = q == 0 ? zx : q == 1 ? zy : q == 2 ? rdzw : rdz;
    wreal* d = q == 0 ? zxd : q == 1 ? zyd : q == 2 ? rdzwd : rdzd;
    v[at] = r.v;
    if (WRFPLUS_TL && tl) d[at] = r.d;
}

extern "C" __global__ void h2_met_ad(const wreal* __restrict__ azx,
                                     const wreal* __restrict__ azy,
                                     const wreal* __restrict__ ardzw,
                                     const wreal* __restrict__ ardz,
                                     const wreal* __restrict__ php,
                                     const wreal* __restrict__ phb, wreal* __restrict__ aphp,
                                     H2_GEOM)
{
    H2Src s = h2_src(H2_GARGS, 0);
    s.php = php; s.phpd = php; s.phb = phb;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int q;
    size_t at;
    wreal acc;
    if (!h2_ad_word(s, H2S_MET, t, azx, azy, ardzw, ardz, &q, &at, &acc)) return;
    aphp[at] = aphp[at] + acc;
}

extern "C" __global__ void h2_rho_tl(const wreal* __restrict__ alt,
                                     const wreal* __restrict__ altd,
                                     const wreal* __restrict__ qv,
                                     const wreal* __restrict__ qvd, wreal* __restrict__ rho,
                                     wreal* __restrict__ rhod, int tl, H2_GEOM)
{
    H2Src s = h2_src(H2_GARGS, tl ? 1 : 2);
    s.alt = alt; s.altd = altd; s.qv = qv; s.qvd = qvd;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nz * ny * nx) return;
    Dl r = h2_rho_ev(s, t);
    rho[t] = r.v;
    if (WRFPLUS_TL && tl) rhod[t] = r.d;
}

extern "C" __global__ void h2_rho_ad(const wreal* __restrict__ arho,
                                     const wreal* __restrict__ alt,
                                     const wreal* __restrict__ qv, wreal* __restrict__ aalt,
                                     wreal* __restrict__ aqv, H2_GEOM)
{
    H2Src s = h2_src(H2_GARGS, 0);
    s.alt = alt; s.altd = alt; s.qv = qv; s.qvd = qv;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int q;
    size_t at;
    wreal acc;
    if (!h2_ad_word(s, H2S_RHO, t, arho, arho, arho, arho, &q, &at, &acc)) return;
    if (q == 0) aalt[at] = aalt[at] + acc;
    else aqv[at] = aqv[at] + acc;
}

extern "C" __global__ void h2_tuv_tl(const wreal* __restrict__ rho,
                                     const wreal* __restrict__ rhod,
                                     const wreal* __restrict__ km,
                                     const wreal* __restrict__ kmd,
                                     const wreal* __restrict__ da,
                                     const wreal* __restrict__ dad,
                                     const wreal* __restrict__ d12,
                                     const wreal* __restrict__ d12d, wreal* __restrict__ ta,
                                     wreal* __restrict__ tad, wreal* __restrict__ t12,
                                     wreal* __restrict__ t12d, int tl, H2_GEOM)
{
    H2Src s = h2_src(H2_GARGS, tl ? 1 : 2);
    s.rho = rho; s.rhod = rhod; s.km = km; s.kmd = kmd;
    s.da = da; s.dad = dad; s.d12 = d12; s.d12d = d12d;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int q;
    size_t at;
    Dl r;
    if (!h2_fwd_word(s, H2S_TUV, t, &q, &at, &r)) return;
    (q == 0 ? ta : t12)[at] = r.v;
    if (WRFPLUS_TL && tl) (q == 0 ? tad : t12d)[at] = r.d;
}

extern "C" __global__ void h2_tuv_ad(const wreal* __restrict__ ata,
                                     const wreal* __restrict__ at12,
                                     const wreal* __restrict__ rho,
                                     const wreal* __restrict__ km,
                                     const wreal* __restrict__ da,
                                     const wreal* __restrict__ d12, wreal* __restrict__ arho,
                                     wreal* __restrict__ akm, wreal* __restrict__ ada,
                                     wreal* __restrict__ ad12, H2_GEOM)
{
    H2Src s = h2_src(H2_GARGS, 0);
    s.rho = rho; s.rhod = rho; s.km = km; s.kmd = km;
    s.da = da; s.dad = da; s.d12 = d12; s.d12d = d12;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int q;
    size_t at;
    wreal acc;
    if (!h2_ad_word(s, H2S_TUV, t, ata, at12, ata, ata, &q, &at, &acc)) return;
    wreal* out = q == 0 ? arho : q == 1 ? akm : q == 2 ? ada : ad12;
    out[at] = out[at] + acc;
}

#define H2_TW_PASSIVE const wreal* __restrict__ msft, const wreal* __restrict__ msfu, \
    const wreal* __restrict__ msfv, const wreal* __restrict__ fnm, const wreal* __restrict__ fnp

extern "C" __global__ void h2_tw_tl(const wreal* __restrict__ u, const wreal* __restrict__ ud,
                                    const wreal* __restrict__ v, const wreal* __restrict__ vd,
                                    const wreal* __restrict__ w, const wreal* __restrict__ wd,
                                    const wreal* __restrict__ zx,
                                    const wreal* __restrict__ zxd,
                                    const wreal* __restrict__ zy,
                                    const wreal* __restrict__ zyd,
                                    const wreal* __restrict__ rdz,
                                    const wreal* __restrict__ rdzd,
                                    const wreal* __restrict__ rho,
                                    const wreal* __restrict__ rhod,
                                    const wreal* __restrict__ km,
                                    const wreal* __restrict__ kmd, H2_TW_PASSIVE,
                                    wreal* __restrict__ t13, wreal* __restrict__ t13d,
                                    wreal* __restrict__ t23, wreal* __restrict__ t23d, int tl,
                                    H2_GEOM)
{
    H2Src s = h2_src(H2_GARGS, tl ? 1 : 2);
    s.u = u; s.ud = ud; s.v = v; s.vd = vd; s.w = w; s.wd = wd;
    s.zx = zx; s.zxd = zxd; s.zy = zy; s.zyd = zyd; s.rz = rdz; s.rzd = rdzd;
    s.rho = rho; s.rhod = rhod; s.km = km; s.kmd = kmd;
    s.msft = msft; s.msfu = msfu; s.msfv = msfv; s.fnm = fnm; s.fnp = fnp;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int q;
    size_t at;
    Dl r;
    if (!h2_fwd_word(s, H2S_TW, t, &q, &at, &r)) return;
    (q == 0 ? t13 : t23)[at] = r.v;
    if (WRFPLUS_TL && tl) (q == 0 ? t13d : t23d)[at] = r.d;
}

extern "C" __global__ void h2_tw_ad(const wreal* __restrict__ at13,
                                    const wreal* __restrict__ at23,
                                    const wreal* __restrict__ u, const wreal* __restrict__ v,
                                    const wreal* __restrict__ w, const wreal* __restrict__ zx,
                                    const wreal* __restrict__ zy,
                                    const wreal* __restrict__ rdz,
                                    const wreal* __restrict__ rho,
                                    const wreal* __restrict__ km, H2_TW_PASSIVE,
                                    wreal* __restrict__ au, wreal* __restrict__ av,
                                    wreal* __restrict__ aw, wreal* __restrict__ azx,
                                    wreal* __restrict__ azy, wreal* __restrict__ ardz,
                                    wreal* __restrict__ arho, wreal* __restrict__ akm,
                                    H2_GEOM)
{
    H2Src s = h2_src(H2_GARGS, 0);
    s.u = u; s.ud = u; s.v = v; s.vd = v; s.w = w; s.wd = w;
    s.zx = zx; s.zxd = zx; s.zy = zy; s.zyd = zy; s.rz = rdz; s.rzd = rdz;
    s.rho = rho; s.rhod = rho; s.km = km; s.kmd = km;
    s.msft = msft; s.msfu = msfu; s.msfv = msfv; s.fnm = fnm; s.fnp = fnp;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int q;
    size_t at;
    wreal acc;
    if (!h2_ad_word(s, H2S_TW, t, at13, at23, at13, at13, &q, &at, &acc)) return;
    wreal* out = q == 0 ? au : q == 1 ? av : q == 2 ? aw : q == 3 ? azx : q == 4 ? azy
                 : q == 5 ? ardz : q == 6 ? arho : akm;
    out[at] = out[at] + acc;
}

#define H2_HS_PASSIVE const wreal* __restrict__ msfu, const wreal* __restrict__ msfv, \
    const wreal* __restrict__ fnm, const wreal* __restrict__ fnp, const wreal* __restrict__ dn, \
    const wreal* __restrict__ dnw

extern "C" __global__ void h2_hs_tl(const wreal* __restrict__ f, const wreal* __restrict__ fd,
                                    const wreal* __restrict__ km,
                                    const wreal* __restrict__ kmd,
                                    const wreal* __restrict__ rho,
                                    const wreal* __restrict__ rhod,
                                    const wreal* __restrict__ zx,
                                    const wreal* __restrict__ zxd,
                                    const wreal* __restrict__ zy,
                                    const wreal* __restrict__ zyd,
                                    const wreal* __restrict__ rdzw,
                                    const wreal* __restrict__ rdzwd, H2_HS_PASSIVE,
                                    wreal* __restrict__ h1, wreal* __restrict__ h1d,
                                    wreal* __restrict__ h2, wreal* __restrict__ h2d, int tl,
                                    H2_GEOM)
{
    H2Src s = h2_src(H2_GARGS, tl ? 1 : 2);
    s.f = f; s.fd = fd; s.km = km; s.kmd = kmd; s.rho = rho; s.rhod = rhod;
    s.zx = zx; s.zxd = zxd; s.zy = zy; s.zyd = zyd; s.rz = rdzw; s.rzd = rdzwd;
    s.msfu = msfu; s.msfv = msfv; s.fnm = fnm; s.fnp = fnp; s.dn = dn; s.dnw = dnw;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int q;
    size_t at;
    Dl r;
    if (!h2_fwd_word(s, H2S_HS, t, &q, &at, &r)) return;
    (q == 0 ? h1 : h2)[at] = r.v;
    if (WRFPLUS_TL && tl) (q == 0 ? h1d : h2d)[at] = r.d;
}

extern "C" __global__ void h2_hs_ad(const wreal* __restrict__ ah1,
                                    const wreal* __restrict__ ah2,
                                    const wreal* __restrict__ f, const wreal* __restrict__ km,
                                    const wreal* __restrict__ rho,
                                    const wreal* __restrict__ zx,
                                    const wreal* __restrict__ zy,
                                    const wreal* __restrict__ rdzw, H2_HS_PASSIVE,
                                    wreal* __restrict__ af, wreal* __restrict__ akm,
                                    wreal* __restrict__ arho, wreal* __restrict__ azx,
                                    wreal* __restrict__ azy, wreal* __restrict__ ardzw,
                                    H2_GEOM)
{
    H2Src s = h2_src(H2_GARGS, 0);
    s.f = f; s.fd = f; s.km = km; s.kmd = km; s.rho = rho; s.rhod = rho;
    s.zx = zx; s.zxd = zx; s.zy = zy; s.zyd = zy; s.rz = rdzw; s.rzd = rdzw;
    s.msfu = msfu; s.msfv = msfv; s.fnm = fnm; s.fnp = fnp; s.dn = dn; s.dnw = dnw;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int q;
    size_t at;
    wreal acc;
    if (!h2_ad_word(s, H2S_HS, t, ah1, ah2, ah1, ah1, &q, &at, &acc)) return;
    wreal* out = q == 0 ? af : q == 1 ? akm : q == 2 ? arho : q == 3 ? azx : q == 4 ? azy
                 : ardzw;
    out[at] = out[at] + acc;
}

#define H2_ROW_PASSIVE const wreal* __restrict__ msft, const wreal* __restrict__ msfu, \
    const wreal* __restrict__ msfv, const wreal* __restrict__ fnm, const wreal* __restrict__ fnp, \
    const wreal* __restrict__ dn, const wreal* __restrict__ dnw

// TL: one thread per point of the row; the tendency accumulates.
extern "C" __global__ void h2_row_tl(wreal* __restrict__ tend, wreal* __restrict__ tendd,
                                     const wreal* __restrict__ s1,
                                     const wreal* __restrict__ s1d,
                                     const wreal* __restrict__ s2,
                                     const wreal* __restrict__ s2d,
                                     const wreal* __restrict__ zx,
                                     const wreal* __restrict__ zxd,
                                     const wreal* __restrict__ zy,
                                     const wreal* __restrict__ zyd,
                                     const wreal* __restrict__ rz,
                                     const wreal* __restrict__ rzd, H2_ROW_PASSIVE, int nl,
                                     H2_GEOM)
{
    H2Src s = h2_src(H2_GARGS, 1);
    s.s1 = s1; s.s1d = s1d; s.s2 = s2; s.s2d = s2d;
    s.zx = zx; s.zxd = zxd; s.zy = zy; s.zyd = zyd; s.rz = rz; s.rzd = rzd;
    s.msft = msft; s.msfu = msfu; s.msfv = msfv; s.fnm = fnm; s.fnp = fnp; s.dn = dn;
    s.dnw = dnw;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int nzs = nz + (row == 3), nys = ny + (row == 2), nxs = nx + (row == 1);
    if (t >= (size_t)nzs * nys * nxs) return;
    int i = (int)(t % nxs), j = (int)((t / nxs) % nys), k = (int)(t / ((size_t)nxs * nys));
    if (row == 3 && (k == 0 || k == nz)) return;   // the primal skips the end levels
    Dl r = h2_eval(s, H2O_TEND, k, j, i);
    if (nl) tend[t] = tend[t] + r.v;
    if (WRFPLUS_TL) tendd[t] = tendd[t] + r.d;
}

// AD of the row: into the stresses' or fluxes' and the metrics' adjoints.
// The tendency adjoint is read, not consumed.
extern "C" __global__ void h2_row_ad(const wreal* __restrict__ atend,
                                     const wreal* __restrict__ s1,
                                     const wreal* __restrict__ s2,
                                     const wreal* __restrict__ zx,
                                     const wreal* __restrict__ zy,
                                     const wreal* __restrict__ rz, H2_ROW_PASSIVE,
                                     wreal* __restrict__ as1, wreal* __restrict__ as2,
                                     wreal* __restrict__ azx, wreal* __restrict__ azy,
                                     wreal* __restrict__ arz, H2_GEOM)
{
    H2Src s = h2_src(H2_GARGS, 0);
    s.s1 = s1; s.s1d = s1; s.s2 = s2; s.s2d = s2;
    s.zx = zx; s.zxd = zx; s.zy = zy; s.zyd = zy; s.rz = rz; s.rzd = rz;
    s.msft = msft; s.msfu = msfu; s.msfv = msfv; s.fnm = fnm; s.fnp = fnp; s.dn = dn;
    s.dnw = dnw;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int q;
    size_t at;
    wreal acc;
    if (!h2_ad_word(s, H2S_ROW, t, atend, atend, atend, atend, &q, &at, &acc)) return;
    wreal* out = q == 0 ? as1 : q == 1 ? as2 : q == 2 ? azx : q == 3 ? azy : arz;
    out[at] = out[at] + acc;
}
