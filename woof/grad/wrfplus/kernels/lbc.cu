// Tangent-linear and adjoint of woof's strict lateral-boundary tendency
// rows (P3a item 7): one row of state_relaxation_multi
// (woof/core/kernels/lbc_state.cu, GPUWM_WRF_EXACT, whole domain), which
// is WRF relax_bdy_dry (relax_bdytend_core on the coupled field) and
// spec_bdy_dry (spec_bdytend) for u, v, theta, phi, w and mu.
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

enum { LB_U = 0, LB_V = 1, LB_THETA = 2, LB_PHI = 3, LB_MU = 4, LB_W = 6 };
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
            heldd[idx] = h.d;
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
    tendd[idx] = result.d;
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
