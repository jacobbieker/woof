// Tangent-linear and adjoint of woof's strict lateral-boundary tendency
// rows (P3a item 7): one row of state_relaxation_multi
// (woof/core/kernels/lbc_state.cu, GPUWM_WRF_EXACT, whole domain), which
// is WRF relax_bdy_dry (relax_bdytend_core on the coupled field) and
// spec_bdy_dry (spec_bdytend) for u, v, theta, phi, w and mu, and qv's
// rk_step-1 capture (relax_bdy_scalar + spec_bdy_scalar), coupled as theta.
//
// Per cell of the row's field, with the row's flags:
//   held pass (held_compute): h = 0 on the specified rows, the relaxation
//     of 0 in the relaxation zone, 0 inside; / msft (divide_msf); stored
//     to the held buffer (held_store);
//   held_read: h = the held buffer;
//   target: with a held pass or read, tend = (spec ? table tendency :
//     tend) + h; otherwise tend = table tendency on the specified rows and
//     (apply_relax) the relaxation of tend in the zone.
// The relaxation is WRF's fcx*f0 - gcx*(f1+f2+f3+f4-4 f0) with
// f = (table value + dtbc * table tendency) - coupled field, at the cell
// and its four neighbours (i or j clamped along the side).  The active
// inputs are the tendency and held rows, the row's field, mu' and the
// eight side tables (value and tendency per side): the LBC perturbation
// is an AD output (4D-Var's var4d_lbc).
//
// The TL's value half repeats the primal's operations in its order
// (--fmad=false; division __fdiv_rn).  The AD is in gather form: a point
// pass writes each frame cell's five stencil adjoints and its specified
// adjoint to scratch, then gathers form the coupled-field adjoint per
// cell, the table adjoints per entry and the mu' adjoint per column (one
// writer per word, no atomics).
//
// Field layout (k, j, i) of the row's own grid (u: nx+1 faces, v: ny+1
// rows, w/phi: nz+1 levels, mu: one level); W/E tables (nz, ny, width),
// S/N tables (nz, width, nx) on that grid; mass columns (mny, mnx).

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

static __device__ __forceinline__ float wdiv(float a, float b) { return __fdiv_rn(a, b); }
static __device__ __forceinline__ double wdiv(double a, double b) { return a / b; }

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
static __device__ __forceinline__ Dl operator+(Dl a, wreal b) { return dl(a.v + b, a.d); }
static __device__ __forceinline__ Dl operator*(Dl a, wreal b) { return dl(a.v * b, a.d * b); }
static __device__ __forceinline__ Dl operator/(Dl a, wreal b)
{
    return dl(wdiv(a.v, b), wdiv(a.d, b));
}
static __device__ __forceinline__ Dl ld(const wreal* a, const wreal* ad, size_t at)
{
    return dl(a[at], ad[at]);
}

// LB_QV takes the mass-grid branches theta takes (c1h coupling, msft).
enum { LB_U = 0, LB_V = 1, LB_THETA = 2, LB_PHI = 3, LB_MU = 4, LB_QV = 5, LB_W = 6 };
enum { SIDE_S = 0, SIDE_N = 1, SIDE_W = 2, SIDE_E = 3 };

// ------------------------------------------------- frame enumeration
// lbc_state.cu's frame_point / frame_offset (whole domain: every ring has
// two rows and two columns).

static __device__ __forceinline__ int lb_min4(int a, int b, int c, int d)
{
    return min(min(a, b), min(c, d));
}

static __device__ __forceinline__ bool lb_frame_point(int p, int ny, int nx, int width,
                                                      int* j, int* i)
{
    for (int d = 0; d < width; ++d) {
        int ni = nx - 2 * d;
        int nj = max(ny - 2 * d - 2, 0);
        int ring = 2 * ni + 2 * nj;
        if (p >= ring) {
            p -= ring;
            continue;
        }
        if (p < ni) {
            *j = d;
            *i = d + p;
        } else if (p < 2 * ni) {
            *j = ny - 1 - d;
            *i = d + p - ni;
        } else if (p < 2 * ni + nj) {
            *j = d + 1 + p - 2 * ni;
            *i = d;
        } else {
            *j = d + 1 + p - 2 * ni - nj;
            *i = nx - 1 - d;
        }
        return true;
    }
    return false;
}

static __device__ __forceinline__ int lb_frame_offset(int j, int i, int ny, int nx, int width)
{
    int d = lb_min4(j, ny - 1 - j, i, nx - 1 - i);
    if (d < 0 || d >= width) return -1;
    int offset = 0;
    for (int q = 0; q < d; ++q) offset += 2 * (nx - 2 * q) + 2 * max(ny - 2 * q - 2, 0);
    int ni = nx - 2 * d;
    int nj = max(ny - 2 * d - 2, 0);
    if (j == d) return offset + i - d;
    if (j == ny - 1 - d) return offset + ni + i - d;
    if (i == d) return offset + 2 * ni + j - d - 1;
    return offset + 2 * ni + nj + j - d - 1;
}

// Thread -> cell: every level's frame rings (width aw) take the first
// nz * ring threads, in lb_frame_point order, and the interior cells the
// rest, so the boundary work runs in dense warps (the primal's remap; a
// row without a held pass launches the frame threads only).
static __device__ __forceinline__ bool lb_cell(size_t tid, int nz, int ny, int nx, int aw,
                                               int* k, int* j, int* i)
{
    size_t iny = (size_t)max(ny - 2 * aw, 0), inx = (size_t)max(nx - 2 * aw, 0);
    size_t ring = (size_t)ny * nx - iny * inx;
    if (tid < (size_t)nz * ring) {
        *k = (int)(tid / ring);
        return lb_frame_point((int)(tid - (size_t)*k * ring), ny, nx, aw, j, i);
    }
    size_t t = tid - (size_t)nz * ring, inner = iny * inx;
    if (inner == 0 || t >= (size_t)nz * inner) return false;
    *k = (int)(t / inner);
    size_t n = t - (size_t)*k * inner;
    *j = aw + (int)(n / inx);
    *i = aw + (int)(n - (size_t)(*j - aw) * inx);
    return true;
}

static __device__ __forceinline__ const wreal* lb_side(int side, const wreal* s,
                                                      const wreal* n, const wreal* w,
                                                      const wreal* e)
{
    return side == 0 ? s : side == 1 ? n : side == 2 ? w : e;
}

// --------------------------------------------------- classification
// 0 interior / none, 1 specified, 2 relaxation; side and distance.  The
// primal's boundary_index, then its four relaxation branches (Y sides own
// the corners).

static __device__ __forceinline__ int lb_class(int j, int i, int ny, int nx, int spec_zone,
                                               int relax_zone, int* side, int* d)
{
    int ds = j, dn = ny - 1 - j, dw = i, de = nx - 1 - i;
    if (ds < spec_zone && i >= ds && i < nx - ds) { *side = SIDE_S; *d = ds; return 1; }
    if (dn < spec_zone && i >= dn && i < nx - dn) { *side = SIDE_N; *d = dn; return 1; }
    if (dw < spec_zone && j >= dw + 1 && j < ny - dw - 1) { *side = SIDE_W; *d = dw; return 1; }
    if (de < spec_zone && j >= de + 1 && j < ny - de - 1) { *side = SIDE_E; *d = de; return 1; }
    if (ds >= spec_zone && ds < relax_zone && i >= ds && i < nx - ds) {
        *side = SIDE_S; *d = ds; return 2;
    }
    if (dn >= spec_zone && dn < relax_zone && i >= dn && i < nx - dn) {
        *side = SIDE_N; *d = dn; return 2;
    }
    if (dw >= spec_zone && dw < relax_zone && j >= dw + 1 && j < ny - dw - 1) {
        *side = SIDE_W; *d = dw; return 2;
    }
    if (de >= spec_zone && de < relax_zone && j >= de + 1 && j < ny - de - 1) {
        *side = SIDE_E; *d = de; return 2;
    }
    return 0;
}

// The relaxation stencil of a zone cell: field cells (cj, ci) and table
// entries (ta, tb) -- (distance, i) on S/N, (j, distance) on W/E -- in
// the primal's f0..f4 order.
static __device__ __forceinline__ void lb_reads(int side, int d, int j, int i, int ny, int nx,
                                                int* cj, int* ci, int* ta, int* tb)
{
    if (side == SIDE_S || side == SIDE_N) {
        int im = max(i - 1, 0), ip = min(i + 1, nx - 1);
        int out = side == SIDE_S ? j - 1 : j + 1;
        int in = side == SIDE_S ? j + 1 : j - 1;
        cj[0] = j;   ci[0] = i;  ta[0] = d;     tb[0] = i;
        cj[1] = j;   ci[1] = im; ta[1] = d;     tb[1] = im;
        cj[2] = j;   ci[2] = ip; ta[2] = d;     tb[2] = ip;
        cj[3] = out; ci[3] = i;  ta[3] = d - 1; tb[3] = i;
        cj[4] = in;  ci[4] = i;  ta[4] = d + 1; tb[4] = i;
    } else {
        int jm = max(j - 1, 0), jp = min(j + 1, ny - 1);
        int out = side == SIDE_W ? i - 1 : i + 1;
        int in = side == SIDE_W ? i + 1 : i - 1;
        cj[0] = j;  ci[0] = i;   ta[0] = j;  tb[0] = d;
        cj[1] = jm; ci[1] = i;   ta[1] = jm; tb[1] = d;
        cj[2] = jp; ci[2] = i;   ta[2] = jp; tb[2] = d;
        cj[3] = j;  ci[3] = out; ta[3] = j;  tb[3] = d - 1;
        cj[4] = j;  ci[4] = in;  ta[4] = j;  tb[4] = d + 1;
    }
}

static __device__ __forceinline__ size_t lb_tidx(int side, int k, int a, int b, int ny, int nx,
                                                 int width)
{
    if (side == SIDE_S || side == SIDE_N)
        return ((size_t)k * width + a) * nx + b;
    return ((size_t)k * ny + a) * width + b;
}

#define LB_TABLES                                                            \
    const wreal* __restrict__ west, const wreal* __restrict__ west_t,        \
    const wreal* __restrict__ east, const wreal* __restrict__ east_t,        \
    const wreal* __restrict__ south, const wreal* __restrict__ south_t,      \
    const wreal* __restrict__ north, const wreal* __restrict__ north_t

#define LB_DTABLES                                                           \
    const wreal* __restrict__ westd, const wreal* __restrict__ west_td,      \
    const wreal* __restrict__ eastd, const wreal* __restrict__ east_td,      \
    const wreal* __restrict__ southd, const wreal* __restrict__ south_td,    \
    const wreal* __restrict__ northd, const wreal* __restrict__ north_td

#define LB_PASSIVE                                                           \
    const wreal* __restrict__ mub, const wreal* __restrict__ c1h,            \
    const wreal* __restrict__ c2h, const wreal* __restrict__ c1f,            \
    const wreal* __restrict__ c2f, const wreal* __restrict__ msft,           \
    const wreal* __restrict__ msfu, const wreal* __restrict__ msfv,          \
    const wreal* __restrict__ fcx, const wreal* __restrict__ gcx

#define LB_GEOM                                                              \
    wreal dtbc, int kind, int width, int spec_zone, int relax_zone, int aw,  \
    int w2, int held_compute, int held_store, int held_read, int target,     \
    int apply_relax, int divide_msf, int has_msf, int nz, int ny, int nx

// ----------------------------------------------------- coupled field
// The primal's coupled_current, strict branches (u_face_mu_wrf,
// v_face_mu_wrf, msfv as the stored inverse, theta' coupled as is).

static __device__ Dl lb_coupled_tl(int kind, int k, int j, int i, const wreal* src,
                                   const wreal* srcd, const wreal* mup, const wreal* mupd,
                                   LB_PASSIVE, int has_msf, int ny, int nx)
{
    int mny = ny - (kind == LB_V);
    int mnx = nx - (kind == LB_U);
    size_t idx = ((size_t)k * ny + j) * nx + i;
    if (kind == LB_MU) return ld(mup, mupd, (size_t)j * mnx + i);
    if (kind == LB_U || kind == LB_V) {
        size_t a, b;
        if (kind == LB_U) {
            a = (size_t)j * mnx + min(i, mnx - 1);
            b = (size_t)j * mnx + max(i - 1, 0);
        } else {
            a = (size_t)min(j, mny - 1) * mnx + i;
            b = (size_t)max(j - 1, 0) * mnx + i;
        }
        Dl mass = (wreal)0.5 * (((ld(mup, mupd, a) + ld(mup, mupd, b)) + mub[a]) + mub[b]);
        Dl ch = c1h[k] * mass + c2h[k];
        Dl r = ch * ld(src, srcd, idx);
        if (!has_msf) return r;
        size_t m = (size_t)j * nx + i;
        return kind == LB_U ? r / msfu[m] : r * msfv[m];
    }
    size_t c = (size_t)j * mnx + i;
    Dl mass = dl(mub[c] + mup[c], mupd[c]);
    Dl ch = (kind == LB_W || kind == LB_PHI) ? c1f[k] * mass + c2f[k]
                                              : c1h[k] * mass + c2h[k];
    return ch * ld(src, srcd, idx);
}

extern "C" __global__
void lbc_tl(wreal* __restrict__ tend, wreal* __restrict__ tendd, wreal* __restrict__ held,
            wreal* __restrict__ heldd, const wreal* __restrict__ src,
            const wreal* __restrict__ srcd, const wreal* __restrict__ mup,
            const wreal* __restrict__ mupd, LB_TABLES, LB_DTABLES, LB_PASSIVE, int nl,
            LB_GEOM)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int k, j, i;
    if (!lb_cell(tid, nz, ny, nx, aw, &k, &j, &i)) return;
    bool perimeter = lb_min4(j, ny - 1 - j, i, nx - 1 - i) < aw;
    if (!perimeter && !(held_compute || held_read)) return;
    size_t idx = ((size_t)k * ny + j) * nx + i;
    int side = 0, d = 0;
    int cls = perimeter ? lb_class(j, i, ny, nx, spec_zone, relax_zone, &side, &d) : 0;
    // the side's tables, picked by value (a pointer array indexed by side
    // would sit in local memory for every thread)
    const wreal* tv = lb_side(side, south, north, west, east);
    const wreal* tt = lb_side(side, south_t, north_t, west_t, east_t);
    const wreal* tvd = lb_side(side, southd, northd, westd, eastd);
    const wreal* ttd = lb_side(side, south_td, north_td, west_td, east_td);
    Dl specified = dl((wreal)0, (wreal)0);
    if (cls == 1) {
        size_t b = side < 2 ? lb_tidx(side, k, d, i, ny, nx, width)
                            : lb_tidx(side, k, j, d, ny, nx, width);
        specified = ld(tt, ttd, b);
    }
    Dl incoming = target ? ld(tend, tendd, idx) : dl((wreal)0, (wreal)0);
    Dl result = held_compute ? dl((wreal)0, (wreal)0) : incoming;
    if (cls == 1) {
        result = held_compute ? dl((wreal)0, (wreal)0) : specified;
    } else if (cls == 2 && (held_compute || apply_relax)) {
        int cj[5], ci[5], ta[5], tb[5];
        lb_reads(side, d, j, i, ny, nx, cj, ci, ta, tb);
        Dl f[5];
        for (int n = 0; n < 5; ++n) {
            size_t b = lb_tidx(side, k, ta[n], tb[n], ny, nx, width);
            Dl value = ld(tv, tvd, b) + dtbc * ld(tt, ttd, b);
            f[n] = value - lb_coupled_tl(kind, k, cj[n], ci[n], src, srcd, mup, mupd, mub,
                                         c1h, c2h, c1f, c2f, msft, msfu, msfv, fcx, gcx,
                                         has_msf, ny, nx);
        }
        Dl sum = ((f[1] + f[2]) + f[3]) + f[4];
        result = (result + fcx[d] * f[0]) - gcx[d] * (sum - (wreal)4 * f[0]);
    }
    Dl h = dl((wreal)0, (wreal)0);
    if (held_compute) {
        if (divide_msf) result = result / msft[(size_t)j * nx + i];
        h = result;
        if (held_store) {
            if (nl) held[idx] = h.v;
            if (WRFPLUS_TL) heldd[idx] = h.d;
        }
    } else if (held_read) {
        h = ld(held, heldd, idx);
    }
    if (!target) return;
    if (held_compute || held_read) {
        result = incoming;
        if (cls == 1) result = specified;
        result = result + h;
    }
    if (nl) tend[idx] = result.v;
    if (WRFPLUS_TL) tendd[idx] = result.d;
}

// --------------------------------------------------------------- AD
// Scratch sf (6, nz, F_aw): per frame cell of the active width, the
// adjoints of f0..f4 and of the specified table tendency.  sm (nz, F_w2):
// per cell of the stencil's width (relax_zone + 1 rings), the adjoint of
// the coupled field's mass (the face mass for u and v).

extern "C" __global__
void lbc_ad_point(wreal* __restrict__ tenda, wreal* __restrict__ helda,
                  wreal* __restrict__ sf, LB_PASSIVE, int nf, LB_GEOM)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int k, j, i;
    if (!lb_cell(tid, nz, ny, nx, aw, &k, &j, &i)) return;
    bool perimeter = lb_min4(j, ny - 1 - j, i, nx - 1 - i) < aw;
    if (!perimeter && !(held_compute || held_read)) return;
    size_t idx = ((size_t)k * ny + j) * nx + i;
    int side = 0, d = 0;
    int cls = perimeter ? lb_class(j, i, ny, nx, spec_zone, relax_zone, &side, &d) : 0;
    wreal A = target ? tenda[idx] : (wreal)0;
    wreal s[6] = {0, 0, 0, 0, 0, 0};
    wreal ar = (wreal)0;
    bool relax = false;
    if (held_compute || held_read) {
        wreal ah = A;
        if (held_compute && held_store) ah += helda[idx];
        if (target) tenda[idx] = cls == 1 ? (wreal)0 : A;
        if (cls == 1 && target) s[5] = A;
        if (held_compute) {
            if (held_store) helda[idx] = (wreal)0;
            if (cls == 2) {
                relax = true;
                ar = divide_msf ? wdiv(ah, msft[(size_t)j * nx + i]) : ah;
            }
        } else {
            helda[idx] += ah;
        }
    } else {
        if (cls == 1) {
            s[5] = A;
            tenda[idx] = (wreal)0;
        } else if (cls == 2 && apply_relax) {
            relax = true;
            ar = A;
        }
    }
    if (relax) {
        s[0] = fcx[d] * ar + (wreal)4 * (gcx[d] * ar);
        wreal g = -(gcx[d] * ar);
        s[1] = g; s[2] = g; s[3] = g; s[4] = g;
    }
    if (!perimeter) return;
    int p = lb_frame_offset(j, i, ny, nx, aw);
    for (int n = 0; n < 6; ++n) sf[((size_t)n * nz + k) * nf + p] = s[n];
}

// Does zone cell P relax, and which of its five stencil cells are (j, i)?
static __device__ __forceinline__ wreal lb_collect_cell(int pj, int pi, int j, int i, int k,
                                                        const wreal* sf, int nf, int spec_zone,
                                                        int relax_zone, int aw, int relaxes,
                                                        int nz, int ny, int nx)
{
    if (pj < 0 || pj >= ny || pi < 0 || pi >= nx) return (wreal)0;
    if (lb_min4(pj, ny - 1 - pj, pi, nx - 1 - pi) >= aw) return (wreal)0;
    int side, d;
    if (!relaxes || lb_class(pj, pi, ny, nx, spec_zone, relax_zone, &side, &d) != 2)
        return (wreal)0;
    int cj[5], ci[5], ta[5], tb[5];
    lb_reads(side, d, pj, pi, ny, nx, cj, ci, ta, tb);
    int p = lb_frame_offset(pj, pi, ny, nx, aw);
    wreal s = (wreal)0;
    for (int n = 0; n < 5; ++n)
        if (cj[n] == j && ci[n] == i) s += sf[((size_t)n * nz + k) * nf + p];
    return s;
}

extern "C" __global__
void lbc_ad_coupled(const wreal* __restrict__ src, const wreal* __restrict__ mup,
                    wreal* __restrict__ srca, const wreal* __restrict__ sf,
                    wreal* __restrict__ sm, LB_PASSIVE, int nf, int n2, LB_GEOM)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)nz * n2) return;
    int k = (int)(tid / n2);
    int q = (int)(tid - (size_t)k * n2);
    int j, i;
    lb_frame_point(q, ny, nx, w2, &j, &i);
    int relaxes = held_compute || apply_relax;
    const int dj[5] = {0, 0, 0, -1, 1};
    const int di[5] = {0, -1, 1, 0, 0};
    wreal s = (wreal)0;
    for (int c = 0; c < 5; ++c)
        s += lb_collect_cell(j + dj[c], i + di[c], j, i, k, sf, nf, spec_zone, relax_zone, aw,
                             relaxes, nz, ny, nx);
    wreal a = -s;
    int mny = ny - (kind == LB_V);
    int mnx = nx - (kind == LB_U);
    size_t idx = ((size_t)k * ny + j) * nx + i;
    wreal am;
    if (kind == LB_MU) {
        am = a;
    } else if (kind == LB_U || kind == LB_V) {
        size_t ca, cb;
        if (kind == LB_U) {
            ca = (size_t)j * mnx + min(i, mnx - 1);
            cb = (size_t)j * mnx + max(i - 1, 0);
        } else {
            ca = (size_t)min(j, mny - 1) * mnx + i;
            cb = (size_t)max(j - 1, 0) * mnx + i;
        }
        wreal mass = (wreal)0.5 * (((mup[ca] + mup[cb]) + mub[ca]) + mub[cb]);
        wreal ch = c1h[k] * mass + c2h[k];
        size_t m = (size_t)j * nx + i;
        wreal t = !has_msf ? a : (kind == LB_U ? wdiv(a, msfu[m]) : a * msfv[m]);
        srca[idx] += ch * t;
        am = c1h[k] * (src[idx] * t);
    } else {
        size_t c = (size_t)j * mnx + i;
        wreal mass = mub[c] + mup[c];
        bool full = kind == LB_W || kind == LB_PHI;
        wreal c1 = full ? c1f[k] : c1h[k];
        wreal ch = c1 * mass + (full ? c2f[k] : c2h[k]);
        srca[idx] += ch * a;
        am = c1 * (src[idx] * a);
    }
    sm[(size_t)k * n2 + q] = am;
}

// One side's table entries: value and tendency adjoints.
extern "C" __global__
void lbc_ad_table(wreal* __restrict__ va, wreal* __restrict__ ta_, const wreal* __restrict__ sf,
                  int side, int nf, LB_GEOM)
{
    bool sn = side == SIDE_S || side == SIDE_N;
    int na = sn ? width : ny, nb = sn ? nx : width;
    size_t per = (size_t)na * nb;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)nz * per) return;
    int k = (int)(tid / per);
    size_t rem = tid - (size_t)k * per;
    int a = (int)(rem / nb), b = (int)(rem - (size_t)a * nb);
    int dd = sn ? a : b;
    if (dd > relax_zone) return;
    int gj, gi;
    if (side == SIDE_S) { gj = dd; gi = b; }
    else if (side == SIDE_N) { gj = ny - 1 - dd; gi = b; }
    else if (side == SIDE_W) { gj = a; gi = dd; }
    else { gj = a; gi = nx - 1 - dd; }
    int relaxes = held_compute || apply_relax;
    const int dj[5] = {0, 0, 0, -1, 1};
    const int di[5] = {0, -1, 1, 0, 0};
    wreal sv = (wreal)0, stt = (wreal)0;
    for (int c = 0; c < 5; ++c) {
        int pj = gj + dj[c], pi = gi + di[c];
        if (pj < 0 || pj >= ny || pi < 0 || pi >= nx) continue;
        if (lb_min4(pj, ny - 1 - pj, pi, nx - 1 - pi) >= aw) continue;
        int ps, d;
        int cls = lb_class(pj, pi, ny, nx, spec_zone, relax_zone, &ps, &d);
        if (ps != side || cls == 0) continue;
        int p = lb_frame_offset(pj, pi, ny, nx, aw);
        if (cls == 1) {
            if (c == 0 && target) stt += sf[((size_t)5 * nz + k) * nf + p];
            continue;
        }
        if (!relaxes) continue;
        int cj[5], ci[5], ra[5], rb[5];
        lb_reads(side, d, pj, pi, ny, nx, cj, ci, ra, rb);
        for (int n = 0; n < 5; ++n) {
            if (ra[n] != a || rb[n] != b) continue;
            wreal g = sf[((size_t)n * nz + k) * nf + p];
            sv += g;
            stt += dtbc * g;
        }
    }
    va[tid] += sv;
    ta_[tid] += stt;
}

// mu' adjoint per mass column: the cells (faces) whose coupled mass reads it.
extern "C" __global__
void lbc_ad_mup(wreal* __restrict__ mupa, const wreal* __restrict__ sm, int n2, LB_GEOM)
{
    int mny = ny - (kind == LB_V);
    int mnx = nx - (kind == LB_U);
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= (size_t)mny * mnx) return;
    int j = (int)(c / mnx), i = (int)(c - (size_t)j * mnx);
    if (lb_min4(j, mny - 1 - j, i, mnx - 1 - i) > w2) return;
    wreal s = (wreal)0;
    if (kind == LB_U || kind == LB_V) {
        for (int q = (kind == LB_U ? i : j); q <= (kind == LB_U ? i : j) + 1; ++q) {
            int last = kind == LB_U ? mnx - 1 : mny - 1;
            int self = kind == LB_U ? i : j;
            int cnt = (min(q, last) == self) + (max(q - 1, 0) == self);
            int fj = kind == LB_U ? j : q, fi = kind == LB_U ? q : i;
            int o = lb_frame_offset(fj, fi, ny, nx, w2);
            if (o < 0 || cnt == 0) continue;
            wreal t = (wreal)0;
            for (int k = 0; k < nz; ++k) t += sm[(size_t)k * n2 + o];
            s += (wreal)0.5 * ((wreal)cnt * t);
        }
    } else {
        int o = lb_frame_offset(j, i, ny, nx, w2);
        if (o >= 0)
            for (int k = 0; k < nz; ++k) s += sm[(size_t)k * n2 + o];
    }
    mupa[c] += s;
}

// ====================================================================
// Item 7b.  WRF spec_bdy_final (woof's strict finalize_state_field /
// finalize_state_multi, the GPUWM_WRF_EXACT branch, and install_mu_boundary
// for mu) and spec_bdyupdate_ph + zero_grad_bdy(w) with the frame's
// calc_p_rho (woof's strict advance_specified_phi_w).
// ====================================================================

static __device__ __forceinline__ Dl operator/(Dl a, Dl b)
{
    wreal v = wdiv(a.v, b.v);
    return dl(v, wdiv(a.d - v * b.d, b.v));
}
static __device__ __forceinline__ Dl operator-(Dl a, wreal b) { return dl(a.v - b, a.d); }
static __device__ __forceinline__ Dl operator+(wreal a, Dl b) { return dl(a + b.v, b.d); }

static __device__ __forceinline__ wreal* lb_side_w(int side, wreal* s, wreal* n, wreal* w,
                                                   wreal* e)
{
    return side == 0 ? s : side == 1 ? n : side == 2 ? w : e;
}

// ------------------------------------------------------ spec_bdy_final
// Specified cells only: target = (xmsf * bfield) / xmu, bfield = table +
// dtbc * table tendency, xmu = c1 * mass + c2 with mass the acoustic
// carrier muts (u, v: the calc_mu_uv_1 face mean 0.5 (r + l)), xmsf msfu,
// msfv or msft for u, v, w (has_msf); mu takes bfield.  Every other cell
// keeps its value.

#define LF_PASSIVE                                                           \
    const wreal* __restrict__ c1h, const wreal* __restrict__ c2h,            \
    const wreal* __restrict__ c1f, const wreal* __restrict__ c2f,            \
    const wreal* __restrict__ msft, const wreal* __restrict__ msfu,          \
    const wreal* __restrict__ msfv

#define LF_GEOM wreal dtbc, int kind, int width, int spec_zone, int has_msf, \
    int nz, int ny, int nx

#define LF_ATABLES                                                           \
    wreal* __restrict__ westa, wreal* __restrict__ west_ta,                  \
    wreal* __restrict__ easta, wreal* __restrict__ east_ta,                  \
    wreal* __restrict__ southa, wreal* __restrict__ south_ta,                \
    wreal* __restrict__ northa, wreal* __restrict__ north_ta

static __device__ __forceinline__ size_t lf_tidx(int side, int k, int j, int i, int d, int ny,
                                                 int nx, int width)
{
    return side < 2 ? lb_tidx(side, k, d, i, ny, nx, width)
                    : lb_tidx(side, k, j, d, ny, nx, width);
}

static __device__ __forceinline__ wreal lf_msf(int kind, int j, int i, int nx, int has_msf,
                                               const wreal* msft, const wreal* msfu,
                                               const wreal* msfv)
{
    if (!has_msf) return (wreal)1;
    size_t c = (size_t)j * nx + i;
    if (kind == LB_U) return msfu[c];
    if (kind == LB_V) return msfv[c];
    if (kind == LB_W) return msft[c];
    return (wreal)1;
}

static __device__ __forceinline__ bool lf_msf_kind(int kind, int has_msf)
{
    return has_msf && (kind == LB_U || kind == LB_V || kind == LB_W);
}

// The cell's coupling mass xmu (value and tangent).
static __device__ __forceinline__ Dl lf_xmu(int kind, int k, int j, int i, const wreal* muts,
                                            const wreal* mutsd, const wreal* c1h,
                                            const wreal* c2h, const wreal* c1f,
                                            const wreal* c2f, int mny, int mnx)
{
    Dl mass;
    if (kind == LB_U) {
        size_t r = (size_t)j * mnx + min(i, mnx - 1);
        size_t l = (size_t)j * mnx + max(i - 1, 0);
        mass = (wreal)0.5 * (ld(muts, mutsd, r) + ld(muts, mutsd, l));
    } else if (kind == LB_V) {
        size_t n = (size_t)min(j, mny - 1) * mnx + i;
        size_t s = (size_t)max(j - 1, 0) * mnx + i;
        mass = (wreal)0.5 * (ld(muts, mutsd, n) + ld(muts, mutsd, s));
    } else {
        mass = ld(muts, mutsd, (size_t)j * mnx + i);
    }
    bool full = kind == LB_W || kind == LB_PHI;
    return (full ? c1f[k] : c1h[k]) * mass + (full ? c2f[k] : c2h[k]);
}

static __device__ __forceinline__ void lf_mass_dims(int kind, int ny, int nx, int* mny,
                                                    int* mnx)
{
    *mny = kind == LB_V ? ny - 1 : ny;
    *mnx = kind == LB_U ? nx - 1 : nx;
}

extern "C" __global__
void lbf_tl(wreal* __restrict__ target, wreal* __restrict__ targetd, LB_TABLES, LB_DTABLES,
            const wreal* __restrict__ muts, const wreal* __restrict__ mutsd, LF_PASSIVE,
            int nl, LF_GEOM)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int k, j, i, side, d;
    if (!lb_cell(tid, nz, ny, nx, spec_zone, &k, &j, &i)) return;
    if (lb_class(j, i, ny, nx, spec_zone, spec_zone, &side, &d) != 1) return;
    int mny, mnx;
    lf_mass_dims(kind, ny, nx, &mny, &mnx);
    size_t b = lf_tidx(side, k, j, i, d, ny, nx, width);
    Dl bf = ld(lb_side(side, south, north, west, east), lb_side(side, southd, northd, westd, eastd), b)
          + dtbc * ld(lb_side(side, south_t, north_t, west_t, east_t),
                      lb_side(side, south_td, north_td, west_td, east_td), b);
    Dl out = bf;
    if (kind != LB_MU) {
        Dl xmu = lf_xmu(kind, k, j, i, muts, mutsd, c1h, c2h, c1f, c2f, mny, mnx);
        Dl num = lf_msf_kind(kind, has_msf)
                     ? lf_msf(kind, j, i, nx, has_msf, msft, msfu, msfv) * bf : bf;
        out = num / xmu;
    }
    size_t idx = ((size_t)k * ny + j) * nx + i;
    if (nl) target[idx] = out.v;
    if (WRFPLUS_TL) targetd[idx] = out.d;
}

// d(xmu) adjoint of a specified cell (0 elsewhere): -a * out / xmu.
static __device__ __forceinline__ wreal lf_axmu(int k, int j, int i, const wreal* targeta,
                                                LB_TABLES, const wreal* muts, LF_PASSIVE,
                                                int mny, int mnx, LF_GEOM)
{
    int side, d;
    if (j < 0 || j >= ny || i < 0 || i >= nx) return (wreal)0;
    if (lb_class(j, i, ny, nx, spec_zone, spec_zone, &side, &d) != 1) return (wreal)0;
    size_t b = lf_tidx(side, k, j, i, d, ny, nx, width);
    wreal bf = lb_side(side, south, north, west, east)[b]
             + dtbc * lb_side(side, south_t, north_t, west_t, east_t)[b];
    Dl xmu = lf_xmu(kind, k, j, i, muts, muts, c1h, c2h, c1f, c2f, mny, mnx);
    wreal num = lf_msf_kind(kind, has_msf)
                    ? lf_msf(kind, j, i, nx, has_msf, msft, msfu, msfv) * bf : bf;
    wreal out = wdiv(num, xmu.v);
    return -targeta[((size_t)k * ny + j) * nx + i] * out / xmu.v;
}

// muts adjoint, one thread per mass column: every specified cell whose
// coupling mass reads it (u, v: the two faces, edge faces counted twice).
extern "C" __global__
void lbf_ad_muts(wreal* __restrict__ mutsa, const wreal* __restrict__ targeta, LB_TABLES,
                 const wreal* __restrict__ muts, LF_PASSIVE, LF_GEOM)
{
    int mny, mnx;
    lf_mass_dims(kind, ny, nx, &mny, &mnx);
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (kind == LB_MU || c >= (size_t)mny * mnx) return;
    int j = (int)(c / mnx), i = (int)(c - (size_t)j * mnx);
    bool full = kind == LB_W || kind == LB_PHI;
    wreal s = (wreal)0;
    for (int k = 0; k < nz; ++k) {
        wreal c1 = full ? c1f[k] : c1h[k];
        if (kind == LB_U || kind == LB_V) {
            for (int q = 0; q < 2; ++q) {
                int fj = kind == LB_V ? j + q : j, fi = kind == LB_U ? i + q : i;
                int self = kind == LB_U ? i : j;
                int f = kind == LB_U ? fi : fj, last = kind == LB_U ? mnx - 1 : mny - 1;
                int cnt = (min(f, last) == self) + (max(f - 1, 0) == self);
                if (cnt == 0) continue;
                wreal a = lf_axmu(k, fj, fi, targeta, west, west_t, east, east_t, south,
                                  south_t, north, north_t, muts, c1h, c2h, c1f, c2f, msft,
                                  msfu, msfv, mny, mnx, dtbc, kind, width, spec_zone, has_msf,
                                  nz, ny, nx);
                s += (wreal)0.5 * (wreal)cnt * (c1 * a);
            }
        } else {
            s += c1 * lf_axmu(k, j, i, targeta, west, west_t, east, east_t, south, south_t,
                              north, north_t, muts, c1h, c2h, c1f, c2f, msft, msfu, msfv,
                              mny, mnx, dtbc, kind, width, spec_zone, has_msf, nz, ny, nx);
        }
    }
    mutsa[c] += s;
}

// Table adjoints, one thread per specified cell (each table entry is read
// by at most one cell); the cell's target adjoint is consumed.  Runs after
// lbf_ad_muts, which reads it.
extern "C" __global__
void lbf_ad_cell(wreal* __restrict__ targeta, LF_ATABLES, const wreal* __restrict__ muts,
                 LF_PASSIVE, LF_GEOM)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int k, j, i, side, d;
    if (!lb_cell(tid, nz, ny, nx, spec_zone, &k, &j, &i)) return;
    if (lb_class(j, i, ny, nx, spec_zone, spec_zone, &side, &d) != 1) return;
    int mny, mnx;
    lf_mass_dims(kind, ny, nx, &mny, &mnx);
    size_t idx = ((size_t)k * ny + j) * nx + i;
    wreal a = targeta[idx];
    wreal abf = a;
    if (kind != LB_MU) {
        Dl xmu = lf_xmu(kind, k, j, i, muts, muts, c1h, c2h, c1f, c2f, mny, mnx);
        abf = a / xmu.v;
        if (lf_msf_kind(kind, has_msf)) abf = abf * lf_msf(kind, j, i, nx, has_msf, msft, msfu, msfv);
    }
    size_t b = lf_tidx(side, k, j, i, d, ny, nx, width);
    lb_side_w(side, southa, northa, westa, easta)[b] += abf;
    lb_side_w(side, south_ta, north_ta, west_ta, east_ta)[b] += dtbc * abf;
    targeta[idx] = (wreal)0;
}

// ------------------------------------- spec_bdyupdate_ph + zero_grad_bdy
// One thread per spec_zone frame column, as advance_specified_phi_w: for
// k = 0..nz, ph'' = (ph'' num) / den + (dtau rph) / den + php (num / den -
// 1) with num = c1f mu_old + c2f, den = c1f muts + c2f, mu_old = muts -
// dtau rmu_t; w'' copies the clamped interior column; then the column's
// calc_p_rho on the new ph'' (diagnose_p_column_muts).

#define SP_GEOM wreal dtau, int spec_zone, int nz, int ny, int nx

static __device__ __forceinline__ bool sp_frame(int j, int i, int ny, int nx, int spec_zone)
{
    return !(i >= spec_zone && i < nx - spec_zone && j >= spec_zone && j < ny - spec_zone);
}

static __device__ __forceinline__ Dl sp_phi(Dl ph, Dl rph, Dl php, Dl mts, Dl mo, wreal c1,
                                            wreal c2, wreal dtau)
{
    Dl num = c1 * mo + c2;
    Dl den = c1 * mts + c2;
    Dl ratio = num / den;
    Dl v0 = (ph * num) / den;
    Dl v2 = (dtau * rph) / (c1 * mts + c2);
    return (v0 + v2) + php * (ratio - (wreal)1);
}

extern "C" __global__
void sph_tl(wreal* __restrict__ ph, wreal* __restrict__ phd, wreal* __restrict__ w,
            wreal* __restrict__ wd, wreal* __restrict__ p, wreal* __restrict__ pd,
            wreal* __restrict__ al, wreal* __restrict__ ald,
            const wreal* __restrict__ rph, const wreal* __restrict__ rphd,
            const wreal* __restrict__ php, const wreal* __restrict__ phpd,
            const wreal* __restrict__ rmu, const wreal* __restrict__ rmud,
            const wreal* __restrict__ muts, const wreal* __restrict__ mutsd,
            const wreal* __restrict__ th, const wreal* __restrict__ thd,
            const wreal* __restrict__ mupp, const wreal* __restrict__ muppd,
            const wreal* __restrict__ thp, const wreal* __restrict__ thpd,
            const wreal* __restrict__ alt, const wreal* __restrict__ altd,
            const wreal* __restrict__ c2a, const wreal* __restrict__ c2ad,
            const wreal* __restrict__ rdnw, const wreal* __restrict__ c1h,
            const wreal* __restrict__ c2h, const wreal* __restrict__ c1f,
            const wreal* __restrict__ c2f, int nl, SP_GEOM)
{
    size_t st = (size_t)ny * nx;
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= st) return;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    if (!sp_frame(j, i, ny, nx, spec_zone)) return;
    Dl mts = ld(muts, mutsd, c);
    Dl mo = mts - dtau * ld(rmu, rmud, c);
    int sj = min(max(j, spec_zone), ny - 1 - spec_zone);
    int si = min(max(i, spec_zone), nx - 1 - spec_zone);
    size_t sc = (size_t)sj * nx + si;
    Dl mu = ld(mupp, muppd, c);
    Dl prev = dl((wreal)0, (wreal)0);
    for (int k = 0; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        Dl phn = sp_phi(ld(ph, phd, f), ld(rph, rphd, f), ld(php, phpd, f), mts, mo, c1f[k],
                        c2f[k], dtau);
        if (nl) {
            ph[f] = phn.v;
            w[f] = w[(size_t)k * st + sc];
        }
        if (WRFPLUS_TL) phd[f] = phn.d;
        if (WRFPLUS_TL) wd[f] = wd[(size_t)k * st + sc];
        if (k > 0) {
            int h = k - 1;
            size_t t = (size_t)h * st + c;
            Dl chm = c1h[h] * mts + c2h[h];
            Dl thr = ld(thp, thpd, t);
            Dl cm = c1h[h] * mu;
            Dl a = ld(alt, altd, t);
            Dl s = a * cm + rdnw[h] * (phn - prev);
            Dl alv = (dl((wreal)-1, (wreal)0) / chm) * s;
            Dl q = (a * (ld(th, thd, t) - cm * thr)) / (chm * ((wreal)300 + thr));
            Dl pv = ld(c2a, c2ad, t) * (q - alv);
            if (nl) {
                al[t] = alv.v;
                p[t] = pv.v;
            }
            if (WRFPLUS_TL) ald[t] = alv.d;
            if (WRFPLUS_TL) pd[t] = pv.d;
        }
        prev = phn;
    }
}

// w'' adjoint of the zero-gradient copy, one thread per (level, interior
// cell): the frame cells whose clamped source it is.  Runs before
// sph_ad_col, which consumes the frame cells' adjoints.
extern "C" __global__
void sph_ad_w(wreal* __restrict__ wa, SP_GEOM)
{
    size_t st = (size_t)ny * nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)(nz + 1) * st) return;
    int k = (int)(tid / st);
    size_t c = tid - (size_t)k * st;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    if (sp_frame(j, i, ny, nx, spec_zone)) return;
    int jlo = j == spec_zone ? 0 : j, jhi = j == ny - 1 - spec_zone ? ny - 1 : j;
    int ilo = i == spec_zone ? 0 : i, ihi = i == nx - 1 - spec_zone ? nx - 1 : i;
    if (jlo == jhi && ilo == ihi) return;
    wreal s = (wreal)0;
    for (int fj = jlo; fj <= jhi; ++fj)
        for (int fi = ilo; fi <= ihi; ++fi)
            if (sp_frame(fj, fi, ny, nx, spec_zone)) s += wa[(size_t)k * st + (size_t)fj * nx + fi];
    wa[tid] += s;
}

extern "C" __global__
void sph_ad_col(wreal* __restrict__ pha, wreal* __restrict__ wa, wreal* __restrict__ pa,
                wreal* __restrict__ ala, wreal* __restrict__ rpha, wreal* __restrict__ phpa,
                wreal* __restrict__ rmua, wreal* __restrict__ mutsa, wreal* __restrict__ tha,
                wreal* __restrict__ muppa, wreal* __restrict__ thpa, wreal* __restrict__ alta,
                wreal* __restrict__ c2aa, const wreal* __restrict__ ph,
                const wreal* __restrict__ rph, const wreal* __restrict__ php,
                const wreal* __restrict__ rmu, const wreal* __restrict__ muts,
                const wreal* __restrict__ th, const wreal* __restrict__ mupp,
                const wreal* __restrict__ thp, const wreal* __restrict__ alt,
                const wreal* __restrict__ c2a, const wreal* __restrict__ rdnw,
                const wreal* __restrict__ c1h, const wreal* __restrict__ c2h,
                const wreal* __restrict__ c1f, const wreal* __restrict__ c2f, SP_GEOM)
{
    size_t st = (size_t)ny * nx;
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= st) return;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    if (!sp_frame(j, i, ny, nx, spec_zone)) return;
    wreal mts = muts[c];
    wreal mo = mts - dtau * rmu[c];
    wreal mu = mupp[c];
    wreal amts = 0, amu = 0;
    // calc_p_rho on the new ph'' (recomputed level by level).
    wreal z = (wreal)0;
    wreal phlo = sp_phi(dl(ph[c], z), dl(rph[c], z), dl(php[c], z), dl(mts, z), dl(mo, z),
                        c1f[0], c2f[0], dtau).v;
    for (int h = 0; h < nz; ++h) {
        size_t t = (size_t)h * st + c, f = (size_t)(h + 1) * st + c;
        wreal phhi = sp_phi(dl(ph[f], z), dl(rph[f], z), dl(php[f], z), dl(mts, z), dl(mo, z),
                            c1f[h + 1], c2f[h + 1], dtau).v;
        wreal chm = c1h[h] * mts + c2h[h];
        wreal thr = thp[t];
        wreal cm = c1h[h] * mu;
        wreal s = alt[t] * cm + rdnw[h] * (phhi - phlo);
        wreal inv = wdiv((wreal)-1, chm);
        wreal alv = inv * s;
        wreal nv = alt[t] * (th[t] - cm * thr);
        wreal dv = chm * ((wreal)300 + thr);
        wreal q = wdiv(nv, dv);
        wreal ap = pa[t];
        wreal aal = ala[t] - ap * c2a[t];
        c2aa[t] += ap * (q - alv);
        wreal aq = ap * c2a[t];
        wreal anv = aq / dv;
        wreal adv = -aq * q / dv;
        wreal achm = adv * ((wreal)300 + thr);
        wreal athr = adv * chm - anv * alt[t] * cm;
        wreal acm = -anv * alt[t] * thr;
        wreal aalt = anv * (th[t] - cm * thr);
        tha[t] += anv * alt[t];
        wreal ainv = aal * s;
        wreal as = aal * inv;
        achm += ainv * inv * inv;
        aalt += as * cm;
        acm += as * alt[t];
        pha[f] += as * rdnw[h];
        pha[t] -= as * rdnw[h];
        amu += acm * c1h[h];
        amts += achm * c1h[h];
        thpa[t] += athr;
        alta[t] += aalt;
        pa[t] = 0;
        ala[t] = 0;
        phlo = phhi;
    }
    // spec_bdyupdate_ph, level by level; the zero-gradient w is consumed.
    wreal amo = 0;
    for (int k = 0; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        wreal c1 = c1f[k], c2 = c2f[k];
        wreal num = c1 * mo + c2, den = c1 * mts + c2;
        wreal ratio = num / den;
        wreal v0 = (ph[f] * num) / den;
        wreal v2 = (dtau * rph[f]) / den;
        wreal a = pha[f];
        phpa[f] += a * (ratio - (wreal)1);
        wreal aratio = a * php[f];
        wreal aden = -a * v2 / den - a * v0 / den - aratio * ratio / den;
        rpha[f] += dtau * (a / den);
        wreal apn = a / den;
        wreal anum = apn * ph[f] + aratio / den;
        pha[f] = apn * num;
        amo += anum * c1;
        amts += aden * c1;
        wa[f] = 0;
    }
    amts += amo;
    rmua[c] += -dtau * amo;
    mutsa[c] += amts;
    muppa[c] += amu;
}
