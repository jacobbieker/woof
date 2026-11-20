// Tangent-linear and adjoint of woof's strict big-step tendencies (P3a
// item 5):
//
//   horizontal_pressure_gradient  woof slow_pgf (woof/core/kernels/
//                    dycore.cu, the GPUWM_WRF_EXACT, _C_BIGSTEP and
//                    _D_DIAGNOSTICS paths: p is the perturbation pressure),
//                    dry (no cqu/cqv), non-hydrostatic, 3-D base state;
//   pg_buoy_w        woof slow_buoyancy, dry (moist_mode 0).
//
// The oracle is WRFPLUS 4.7.1 G_/A_HORIZONTAL_PRESSURE_GRADIENT and
// g_/a_pg_buoy_w; see docs/dev/P3A-WRF-DRYCORE-ADJOINT.md.
//
// wreal is binary64 unless the unit is built with -DWRFPLUS_REAL=float.
// The TL kernels are the strict primal statements on a value/tangent pair
// (Dl): the value half repeats the primal's operations in its order, so
// the float instantiation's base state gives the primal's words.  Plain
// '+', '-', '*' are the primal's __f*_rn only with --fmad=false, which the
// wrapper passes; division is __fdiv_rn.
//
// Both operators ADD into the momentum tendencies, so the tendencies'
// adjoints pass through and the AD only adds into the inputs' adjoints.
// The AD is in gather form: one thread per input word, which collects the
// faces (levels) that read it (one writer per word, no atomics).
//
// Layout (k, j, i): mass (nz, ny, nx), u (nz, ny, nx+1), v (nz, ny+1, nx),
// w levels (nz+1, ny, nx), columns (ny, nx).

#ifndef WRFPLUS_REAL
#define WRFPLUS_REAL double
#endif
typedef WRFPLUS_REAL wreal;

#define R3(k, j, i, nyy, nxx) \
    ((size_t)(k) * (size_t)(nyy) * (size_t)(nxx) + (size_t)(j) * (size_t)(nxx) + (size_t)(i))

#define BS_G ((wreal)9.81)

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
static __device__ __forceinline__ Dl operator-(Dl a, wreal b) { return dl(a.v - b, a.d); }
static __device__ __forceinline__ Dl ld(const wreal* a, const wreal* ad, size_t at)
{
    return dl(a[at], ad[at]);
}

// ======================================== horizontal_pressure_gradient

#define PGF_GEOM wreal rdx, wreal rdy, wreal cf1, wreal cf2, wreal cf3, \
    wreal cfn, wreal cfn1, int top_lid, int bnd_x, int bnd_y, int nz, int ny, int nx

#define PGF_PASSIVE                                                         \
    const wreal* __restrict__ pb, const wreal* __restrict__ phb,            \
    const wreal* __restrict__ mub, const wreal* __restrict__ c1h,           \
    const wreal* __restrict__ c2h, const wreal* __restrict__ rdnw,          \
    const wreal* __restrict__ fnm, const wreal* __restrict__ fnp

// The primal's wrf_dpn_face: the face-paired p' at interface kf.
static __device__ Dl pgf_dpn_tl(const wreal* p, const wreal* pd, int kf, size_t ca,
                                size_t cb, size_t st, int nz, const wreal* fnm,
                                const wreal* fnp, wreal cf1, wreal cf2, wreal cf3,
                                int top_lid, wreal cfn, wreal cfn1)
{
    if (kf == nz && !top_lid) return dl((wreal)0, (wreal)0);
    int k0 = kf == nz ? nz - 1 : kf;
    Dl pair0 = ld(p, pd, (size_t)k0 * st + cb) + ld(p, pd, (size_t)k0 * st + ca);
    if (kf == 0) {
        Dl pair1 = ld(p, pd, st + cb) + ld(p, pd, st + ca);
        Dl pair2 = ld(p, pd, 2 * st + cb) + ld(p, pd, 2 * st + ca);
        return (wreal)0.5 * ((cf1 * pair0 + cf2 * pair1) + cf3 * pair2);
    }
    int k1 = kf == nz ? nz - 2 : kf - 1;
    Dl pair1 = ld(p, pd, (size_t)k1 * st + cb) + ld(p, pd, (size_t)k1 * st + ca);
    Dl high = (kf == nz ? cfn : fnm[kf]) * pair0;
    Dl low = (kf == nz ? cfn1 : fnp[kf]) * pair1;
    return (wreal)0.5 * (high + low);
}

static __device__ Dl pgf_php_half_tl(const wreal* php, const wreal* phpd, const wreal* phb,
                                     int k, size_t c, size_t st)
{
    size_t a = (size_t)k * st + c;
    wreal base = phb[a] + phb[a + st];
    return (wreal)0.5 * ((ld(php, phpd, a) + base) + ld(php, phpd, a + st));
}

// The primal's pgf_face (strict): the term the face's tendency loses.
static __device__ Dl pgf_face_tl(size_t ca, size_t cb, int k, size_t st, int nz, wreal rd,
                                 const wreal* p, const wreal* pd, const wreal* al,
                                 const wreal* ald, const wreal* alt, const wreal* altd,
                                 const wreal* php, const wreal* phpd, const wreal* mup,
                                 const wreal* mupd, PGF_PASSIVE, wreal cf1, wreal cf2,
                                 wreal cf3, int top_lid, wreal cfn, wreal cfn1)
{
    size_t a = (size_t)k * st + ca, b = (size_t)k * st + cb;
    wreal half_rd = (wreal)0.5 * rd;
    Dl muf = (wreal)0.5 * (((ld(mup, mupd, ca) + ld(mup, mupd, cb)) + mub[ca]) + mub[cb]);
    Dl layer = c1h[k] * muf + c2h[k];
    Dl dph_hi = ld(php, phpd, a + st) - ld(php, phpd, b + st);
    Dl bracket = (dph_hi + ld(php, phpd, a)) - ld(php, phpd, b);
    Dl alt_sum = ld(alt, altd, a) + ld(alt, altd, b);
    Dl dpp = ld(p, pd, a) - ld(p, pd, b);
    bracket = bracket + alt_sum * dpp;
    Dl al_sum = ld(al, ald, a) + ld(al, ald, b);
    wreal dpb = pb[a] - pb[b];
    bracket = bracket + al_sum * dl(dpb, (wreal)0);
    Dl left = (half_rd * layer) * bracket;
    Dl dphp = pgf_php_half_tl(php, phpd, phb, k, ca, st) - pgf_php_half_tl(php, phpd, phb, k, cb, st);
    Dl hi = pgf_dpn_tl(p, pd, k + 1, ca, cb, st, nz, fnm, fnp, cf1, cf2, cf3, top_lid, cfn, cfn1);
    Dl lo = pgf_dpn_tl(p, pd, k, ca, cb, st, nz, fnm, fnp, cf1, cf2, cf3, top_lid, cfn, cfn1);
    Dl vertical = rdnw[k] * (hi - lo);
    Dl pmass = (wreal)0.5 * (c1h[k] * ld(mup, mupd, cb) + c1h[k] * ld(mup, mupd, ca));
    vertical = vertical - pmass;
    Dl right = (rd * dphp) * vertical;
    return left + right;
}

extern "C" __global__
void pgf_tl(wreal* __restrict__ ru_t, wreal* __restrict__ ru_d,
            wreal* __restrict__ rv_t, wreal* __restrict__ rv_d,
            const wreal* __restrict__ p, const wreal* __restrict__ pd,
            const wreal* __restrict__ al, const wreal* __restrict__ ald,
            const wreal* __restrict__ alt, const wreal* __restrict__ altd,
            const wreal* __restrict__ php, const wreal* __restrict__ phpd,
            const wreal* __restrict__ mup, const wreal* __restrict__ mupd,
            PGF_PASSIVE, int nl, PGF_GEOM)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    size_t lay = (size_t)nyf * nxf;
    if (tid >= (size_t)nz * lay) return;
    int k = (int)(tid / lay);
    int rem = (int)(tid - (size_t)k * lay);
    int j = rem / nxf;
    int i = rem - j * nxf;
    size_t st = (size_t)ny * nx;
    if (j < ny && ((!bnd_x && i <= nx) || (bnd_x && i > 0 && i < nx))) {
        int ia = i % nx, ib = (i - 1 + nx) % nx;
        Dl t = pgf_face_tl((size_t)j * nx + ia, (size_t)j * nx + ib, k, st, nz, rdx,
                           p, pd, al, ald, alt, altd, php, phpd, mup, mupd,
                           pb, phb, mub, c1h, c2h, rdnw, fnm, fnp,
                           cf1, cf2, cf3, top_lid, cfn, cfn1);
        size_t out = R3(k, j, i, ny, nxf);
        if (nl) ru_t[out] = ru_t[out] - t.v;
        ru_d[out] = ru_d[out] - t.d;
    }
    if (i < nx && ((!bnd_y && j <= ny) || (bnd_y && j > 0 && j < ny))) {
        int ja = j % ny, jb = (j - 1 + ny) % ny;
        Dl t = pgf_face_tl((size_t)ja * nx + i, (size_t)jb * nx + i, k, st, nz, rdy,
                           p, pd, al, ald, alt, altd, php, phpd, mup, mupd,
                           pb, phb, mub, c1h, c2h, rdnw, fnm, fnp,
                           cf1, cf2, cf3, top_lid, cfn, cfn1);
        size_t out = R3(k, j, i, nyf, nx);
        if (nl) rv_t[out] = rv_t[out] - t.v;
        rv_d[out] = rv_d[out] - t.d;
    }
}

// ------------------------------------------------------------------ AD
//
// term = (H L) B + (R D) V with H = rd/2, L the layer mass, B the bracket,
// R = rd, D the half-level geopotential difference and V the vertical
// term; the face's tendency loses it, so with a its adjoint
// aB = -a H L, aL = -a H B, aD = -a R V, aV = -a R D.
//
// pgf_ad_face writes the four per face (slot s of fu / fv: aB, aL, aD,
// aV; zero where the face does not exist); pgf_ad_cell (p, al, alt),
// pgf_ad_php and pgf_ad_mup then gather them.

static __device__ __forceinline__ int pgf_face_of(int r, int pc, int n, int bnd)
{
    // One face of an axis that reads a cell: r = 0 as its a cell (face
    // pc), 1 as a of the periodic end face n, 2 as its b cell (face pc+1),
    // 3 as b of the periodic face 0.  -1 when that face does not exist.
    if (r == 0) return (bnd ? pc >= 1 : 1) ? pc : -1;
    if (r == 1) return (!bnd && pc == 0) ? n : -1;
    if (r == 2) return (bnd ? pc + 1 <= n - 1 : 1) ? pc + 1 : -1;
    return (!bnd && pc == n - 1) ? 0 : -1;
}

struct PgfFace {
    size_t ca, cb;      // the face's a (east/north) and b cells, as columns
    size_t aix;         // face index at level 0
    size_t astride;     // face level stride
    int dir;            // 0: u faces, 1: v faces
};

static __device__ __forceinline__ PgfFace pgf_face(int dir, int f, int cross, int ny, int nx)
{
    PgfFace F;
    F.dir = dir;
    if (dir == 0) {
        int ia = f % nx, ib = (f - 1 + nx) % nx;
        F.ca = (size_t)cross * nx + ia;
        F.cb = (size_t)cross * nx + ib;
        F.aix = R3(0, cross, f, ny, nx + 1);
        F.astride = (size_t)ny * (nx + 1);
    } else {
        int ja = f % ny, jb = (f - 1 + ny) % ny;
        F.ca = (size_t)ja * nx + cross;
        F.cb = (size_t)jb * nx + cross;
        F.aix = R3(0, f, cross, ny + 1, nx);
        F.astride = (size_t)(ny + 1) * nx;
    }
    return F;
}

// Slot s of face F at level q.
static __device__ __forceinline__ wreal pgf_fa(PgfFace F, const wreal* fu, const wreal* fv,
                                               size_t nu, size_t nv, int s, int q)
{
    size_t at = F.aix + (size_t)q * F.astride;
    return F.dir ? fv[(size_t)s * nv + at] : fu[(size_t)s * nu + at];
}

// The value half of pgf_face_tl, split into its factors.
struct PgfParts { wreal L, B, D, V; };

static __device__ PgfParts pgf_face_parts(PgfFace F, int k, size_t st, int nz, const wreal* p,
                                      const wreal* al, const wreal* alt, const wreal* php,
                                      const wreal* mup, PGF_PASSIVE, wreal cf1, wreal cf2,
                                      wreal cf3, int top_lid, wreal cfn, wreal cfn1)
{
    PgfParts R;
    size_t a = (size_t)k * st + F.ca, b = (size_t)k * st + F.cb;
    wreal muf = (wreal)0.5 * (((mup[F.ca] + mup[F.cb]) + mub[F.ca]) + mub[F.cb]);
    R.L = c1h[k] * muf + c2h[k];
    wreal br = ((php[a + st] - php[b + st]) + php[a]) - php[b];
    br = br + (alt[a] + alt[b]) * (p[a] - p[b]);
    R.B = br + (al[a] + al[b]) * (pb[a] - pb[b]);
    wreal ha = (wreal)0.5 * (((phb[a] + phb[a + st]) + php[a]) + php[a + st]);
    wreal hb = (wreal)0.5 * (((phb[b] + phb[b + st]) + php[b]) + php[b + st]);
    R.D = ha - hb;
    wreal dpn[2];
    for (int h = 0; h < 2; ++h) {
        int kf = k + h;
        wreal v = 0;
        if (!(kf == nz && !top_lid)) {
            int k0 = kf == nz ? nz - 1 : kf;
            wreal pair0 = p[(size_t)k0 * st + F.cb] + p[(size_t)k0 * st + F.ca];
            if (kf == 0) {
                wreal pair1 = p[st + F.cb] + p[st + F.ca];
                wreal pair2 = p[2 * st + F.cb] + p[2 * st + F.ca];
                v = (wreal)0.5 * ((cf1 * pair0 + cf2 * pair1) + cf3 * pair2);
            } else {
                int k1 = kf == nz ? nz - 2 : kf - 1;
                wreal pair1 = p[(size_t)k1 * st + F.cb] + p[(size_t)k1 * st + F.ca];
                v = (wreal)0.5 * ((kf == nz ? cfn : fnm[kf]) * pair0
                                  + (kf == nz ? cfn1 : fnp[kf]) * pair1);
            }
        }
        dpn[h] = v;
    }
    R.V = rdnw[k] * (dpn[1] - dpn[0]) - (wreal)0.5 * (c1h[k] * mup[F.cb] + c1h[k] * mup[F.ca]);
    return R;
}

#define PGF_AD_IN                                                           \
    const wreal* __restrict__ p, const wreal* __restrict__ al,             \
    const wreal* __restrict__ alt, const wreal* __restrict__ php,          \
    const wreal* __restrict__ mup

extern "C" __global__
void pgf_ad_face(const wreal* __restrict__ ru_a, const wreal* __restrict__ rv_a, PGF_AD_IN,
                 wreal* __restrict__ fu, wreal* __restrict__ fv, PGF_PASSIVE, PGF_GEOM)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    size_t lay = (size_t)nyf * nxf;
    if (tid >= (size_t)nz * lay) return;
    int k = (int)(tid / lay);
    int rem = (int)(tid - (size_t)k * lay);
    int j = rem / nxf;
    int i = rem - j * nxf;
    size_t st = (size_t)ny * nx;
    size_t nu = (size_t)nz * ny * nxf, nv = (size_t)nz * nyf * nx;
    for (int dir = 0; dir < 2; ++dir) {
        if (dir == 0 ? !(j < ny) : !(i < nx)) continue;
        int f = dir ? j : i, n = dir ? ny : nx, bnd = dir ? bnd_y : bnd_x;
        PgfFace F = pgf_face(dir, f, dir ? i : j, ny, nx);
        size_t at = F.aix + (size_t)k * F.astride;
        wreal* out = dir ? fv : fu;
        size_t ns = dir ? nv : nu;
        wreal aB = 0, aL = 0, aD = 0, aV = 0;
        if (bnd ? (f > 0 && f < n) : 1) {
            wreal a = dir ? rv_a[at] : ru_a[at];
            wreal rd = dir ? rdy : rdx;
            PgfParts R = pgf_face_parts(F, k, st, nz, p, al, alt, php, mup, pb, phb, mub, c1h,
                                        c2h, rdnw, fnm, fnp, cf1, cf2, cf3, top_lid, cfn, cfn1);
            wreal hr = -a * ((wreal)0.5 * rd);
            aB = hr * R.L;
            aL = hr * R.B;
            aD = -a * rd * R.V;
            aV = -a * rd * R.D;
        }
        out[at] = aB;
        out[ns + at] = aL;
        out[2 * ns + at] = aD;
        out[3 * ns + at] = aV;
    }
}

#define PGF_GATHER_ARGS                                                     \
    const wreal* __restrict__ fu, const wreal* __restrict__ fv, PGF_AD_IN, \
    wreal* __restrict__ ap, wreal* __restrict__ aal,                        \
    wreal* __restrict__ aalt, wreal* __restrict__ aphp,                     \
    wreal* __restrict__ amup, PGF_PASSIVE, PGF_GEOM

// Weight of p at level m in the face-paired interface value dpn(kf)
// (before its 1/2).
static __device__ __forceinline__ wreal pgf_dpn_w(int kf, int m, int nz, const wreal* fnm,
                                                  const wreal* fnp, wreal cf1, wreal cf2,
                                                  wreal cf3, int top_lid, wreal cfn, wreal cfn1)
{
    if (kf == 0) return m == 0 ? cf1 : (m == 1 ? cf2 : (m == 2 ? cf3 : (wreal)0));
    if (kf == nz) {
        if (!top_lid) return (wreal)0;
        return m == nz - 1 ? cfn : (m == nz - 2 ? cfn1 : (wreal)0);
    }
    return m == kf ? fnm[kf] : (m == kf - 1 ? fnp[kf] : (wreal)0);
}

// p, al and alt at mass point (k, j, i).
extern "C" __global__ void pgf_ad_cell(PGF_GATHER_ARGS)
{
    size_t st = (size_t)ny * nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)nz * st) return;
    int k = (int)(tid / st);
    size_t c = tid - (size_t)k * st;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    size_t nu = (size_t)nz * ny * (nx + 1), nv = (size_t)nz * (ny + 1) * nx;
    wreal gp = 0, gal = 0, galt = 0;
    for (int dir = 0; dir < 2; ++dir) {
        int n = dir ? ny : nx, pc = dir ? j : i, bnd = dir ? bnd_y : bnd_x;
        for (int r = 0; r < 4; ++r) {
            int f = pgf_face_of(r, pc, n, bnd);
            if (f < 0) continue;
            wreal sg = r < 2 ? (wreal)1 : (wreal)-1;
            PgfFace F = pgf_face(dir, f, dir ? i : j, ny, nx);
            size_t a = (size_t)k * st + F.ca, b = (size_t)k * st + F.cb;
            wreal aB = pgf_fa(F, fu, fv, nu, nv, 0, k);
            galt += aB * (p[a] - p[b]);
            gal += aB * (pb[a] - pb[b]);
            gp += sg * aB * (alt[a] + alt[b]);
            // p through the interface values dpn(kf) of the vertical term,
            // V(q) = rdnw[q] (dpn(q+1) - dpn(q)) - ...
            for (int u = 0; u < 4; ++u) {
                int kf = u == 0 ? 0 : (u == 1 ? k : (u == 2 ? k + 1 : nz));
                if (u == 0 && k > 2) continue;
                if (u == 1 && !(k >= 1 && k <= nz - 1)) continue;
                if (u == 2 && !(k + 1 <= nz - 1)) continue;
                wreal w = pgf_dpn_w(kf, k, nz, fnm, fnp, cf1, cf2, cf3, top_lid, cfn, cfn1);
                if (w == (wreal)0) continue;
                wreal adpn = 0;
                if (kf - 1 >= 0) adpn += pgf_fa(F, fu, fv, nu, nv, 3, kf - 1) * rdnw[kf - 1];
                if (kf <= nz - 1) adpn -= pgf_fa(F, fu, fv, nu, nv, 3, kf) * rdnw[kf];
                gp += (wreal)0.5 * w * adpn;
            }
        }
    }
    size_t me = (size_t)k * st + c;
    ap[me] += gp;
    aal[me] += gal;
    aalt[me] += galt;
}

// php (the w-level perturbation geopotential) at (m, j, i), m = 0..nz.
extern "C" __global__ void pgf_ad_php(PGF_GATHER_ARGS)
{
    size_t st = (size_t)ny * nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)(nz + 1) * st) return;
    int m = (int)(tid / st);
    size_t c = tid - (size_t)m * st;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    size_t nu = (size_t)nz * ny * (nx + 1), nv = (size_t)nz * (ny + 1) * nx;
    wreal g = 0;
    for (int dir = 0; dir < 2; ++dir) {
        int n = dir ? ny : nx, pc = dir ? j : i, bnd = dir ? bnd_y : bnd_x;
        for (int r = 0; r < 4; ++r) {
            int f = pgf_face_of(r, pc, n, bnd);
            if (f < 0) continue;
            wreal sg = r < 2 ? (wreal)1 : (wreal)-1;
            PgfFace F = pgf_face(dir, f, dir ? i : j, ny, nx);
            for (int q = m - 1; q <= m; ++q) {
                if (q < 0 || q > nz - 1) continue;
                g += sg * (pgf_fa(F, fu, fv, nu, nv, 0, q)
                           + (wreal)0.5 * pgf_fa(F, fu, fv, nu, nv, 2, q));
            }
        }
    }
    aphp[tid] += g;
}

// mup at column (j, i).
extern "C" __global__ void pgf_ad_mup(PGF_GATHER_ARGS)
{
    size_t st = (size_t)ny * nx;
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= st) return;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    size_t nu = (size_t)nz * ny * (nx + 1), nv = (size_t)nz * (ny + 1) * nx;
    wreal g = 0;
    for (int dir = 0; dir < 2; ++dir) {
        int n = dir ? ny : nx, pc = dir ? j : i, bnd = dir ? bnd_y : bnd_x;
        for (int r = 0; r < 4; ++r) {
            int f = pgf_face_of(r, pc, n, bnd);
            if (f < 0) continue;
            PgfFace F = pgf_face(dir, f, dir ? i : j, ny, nx);
            for (int q = 0; q < nz; ++q)
                g += (wreal)0.5 * c1h[q] * (pgf_fa(F, fu, fv, nu, nv, 1, q)
                                            - pgf_fa(F, fu, fv, nu, nv, 3, q));
        }
    }
    amup[c] += g;
}

// ============================================================ pg_buoy_w

#define BUOY_PASSIVE                                                        \
    const wreal* __restrict__ rdn, const wreal* __restrict__ rdnw,          \
    const wreal* __restrict__ c1f, const wreal* __restrict__ msft

// The primal's buoyancy_scale: (1/msft) g, left to right, or g.
static __device__ __forceinline__ wreal buoy_scale(const wreal* msft, size_t c, int has_msf)
{
    if (!has_msf) return BS_G;
    return wdiv((wreal)1, msft[c]) * BS_G;
}

extern "C" __global__
void buoy_tl(wreal* __restrict__ rw_t, wreal* __restrict__ rw_d,
             const wreal* __restrict__ p, const wreal* __restrict__ pd,
             const wreal* __restrict__ mup, const wreal* __restrict__ mupd,
             BUOY_PASSIVE, int nl, int has_msf, int nz, int ny, int nx)
{
    size_t st = (size_t)ny * nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)nz * st) return;
    int k = (int)(tid / st) + 1;
    size_t c = tid - (size_t)(k - 1) * st;
    size_t ix = (size_t)k * st + c;
    Dl term;
    if (k == nz) {
        wreal cq1 = wdiv((wreal)1, (wreal)1 + (wreal)0);
        Dl pressure = ((cq1 * (wreal)2) * rdnw[nz - 1]) * (dl((wreal)0, (wreal)0)
                                                          - ld(p, pd, (size_t)(nz - 1) * st + c));
        term = pressure - c1f[nz] * ld(mup, mupd, c);
    } else {
        Dl dp = ld(p, pd, ix) - ld(p, pd, ix - st);
        term = rdn[k] * dp;
        term = term - c1f[k] * ld(mup, mupd, c);
    }
    Dl buoy = buoy_scale(msft, c, has_msf) * term;
    if (nl) rw_t[ix] = rw_t[ix] + buoy.v;
    rw_d[ix] = rw_d[ix] + buoy.d;
}

// p and mup of one column.
extern "C" __global__
void buoy_ad(const wreal* __restrict__ rw_a, wreal* __restrict__ ap,
             wreal* __restrict__ amup, BUOY_PASSIVE, int has_msf, int nz, int ny, int nx)
{
    size_t st = (size_t)ny * nx;
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= st) return;
    wreal s = buoy_scale(msft, c, has_msf);
    wreal gm = 0;
    for (int k = nz; k >= 1; --k) {
        wreal a = rw_a[(size_t)k * st + c] * s;
        gm -= a * c1f[k];
        if (k == nz) {
            ap[(size_t)(nz - 1) * st + c] -= a * ((wreal)2 * rdnw[nz - 1]);
        } else {
            ap[(size_t)k * st + c] += a * rdn[k];
            ap[(size_t)(k - 1) * st + c] -= a * rdn[k];
        }
    }
    amup[c] += gm;
}
