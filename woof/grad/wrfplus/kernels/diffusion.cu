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
    s.nz = nz; s.ny = ny; s.nx = nx; s.bx = bx; s.by = by;
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
// The outputs that can read a word lie within 2 columns and rows of it
// and at the levels L-1..L+1, 0 or nz-1 (the bottom and lid extrapolations
// read levels 0-2 and nz-2..nz-1); df_cand drops repeats.
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
    int levels[5] = {k - 1, k, k + 1, 0, nz - 1};
    wreal acc = (wreal)0;
    for (int a = 0; a < 5; ++a) {
        int ko = levels[a];
        if (ko < 0 || ko >= nz) continue;
        bool seen = false;
        for (int b = 0; b < a; ++b) seen = seen || levels[b] == ko;
        if (seen) continue;
        for (int oj = -2; oj <= 2; ++oj) {
            int jo = df_cand(j, oj, -2, ny, by);
            if (jo < 0) continue;
            for (int oi = -2; oi <= 2; ++oi) {
                int io = df_cand(i, oi, -2, nx, bx);
                if (io < 0) continue;
                size_t o = R3(ko, jo, io, ny, nx);
                // defor11 does not read v, nor defor22 u
                if (a11[o] != (wreal)0 && s.which != DF_V)
                    acc = acc + a11[o] * df_d11(s, ko, jo, io).d;
                if (a22[o] != (wreal)0 && s.which != DF_U)
                    acc = acc + a22[o] * df_d22(s, ko, jo, io).d;
                if (a12[o] != (wreal)0) acc = acc + a12[o] * df_d12(s, ko, jo, io).d;
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
