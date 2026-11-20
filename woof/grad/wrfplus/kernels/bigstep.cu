// Tangent-linear and adjoint of woof's strict big-step tendencies (P3a
// item 5):
//
//   horizontal_pressure_gradient  woof slow_pgf (woof/core/kernels/
//                    dycore.cu, the GPUWM_WRF_EXACT, _C_BIGSTEP and
//                    _D_DIAGNOSTICS paths: p is the perturbation pressure),
//                    dry (no cqu/cqv), non-hydrostatic, 3-D base state;
//   pg_buoy_w        woof slow_buoyancy, dry (moist_mode 0).
//   rhs_ph           woof slow_geopotential_open (supplied = 0,
//                    add_vertical = 1), which for h_sca_adv_order 5 on
//                    periodic or specified axes gives the words of the
//                    fused slow_geopotential; 3-D base state.
//   coriolis + curvature  woof coriolis_curvature (the
//                    GPUWM_WRF_EXACT_C_BIGSTEP path), map_proj != 6.
//   w_damp           woof w_damp (openbc.cu), w_damping = 1.
//
// The oracle is WRFPLUS 4.7.1 G_/A_HORIZONTAL_PRESSURE_GRADIENT,
// g_/a_pg_buoy_w, g_/a_rhs_ph, G_/A_CORIOLIS + G_/A_CURVATURE and
// g_/a_w_damp; see
// docs/dev/P3A-WRF-DRYCORE-ADJOINT.md.
//
// wreal is binary64 unless the unit is built with -DWRFPLUS_REAL=float.
// The TL kernels are the strict primal statements on a value/tangent pair
// (Dl): the value half repeats the primal's operations in its order, so
// the float instantiation's base state gives the primal's words.  Plain
// '+', '-', '*' are the primal's __f*_rn only with --fmad=false, which the
// wrapper passes; division is __fdiv_rn.
//
// The pressure gradient and buoyancy ADD into the momentum tendencies,
// so the tendencies' adjoints pass through and the AD only adds into the
// inputs' adjoints.  rhs_ph adds too, except at the top w level, which
// it overwrites: there the tendency's adjoint is consumed and zeroed.
// The AD is in gather form: one thread per input word, which collects the
// faces (levels) that read it (one writer per word, no atomics).
//
// Layout (k, j, i): mass (nz, ny, nx), u (nz, ny, nx+1), v (nz, ny+1, nx),
// w levels (nz+1, ny, nx), columns (ny, nx).

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
static __device__ __forceinline__ Dl operator*(Dl a, wreal b) { return dl(a.v * b, a.d * b); }
static __device__ __forceinline__ Dl operator/(Dl a, wreal b)
{
    return dl(wdiv(a.v, b), wdiv(a.d, b));
}
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
    size_t rem = tid - (size_t)k * lay;
    int j = (int)(rem / nxf);
    int i = (int)(rem - (size_t)j * nxf);
    size_t st = (size_t)ny * nx;
    if (j < ny && ((!bnd_x && i <= nx) || (bnd_x && i > 0 && i < nx))) {
        int ia = i % nx, ib = (i - 1 + nx) % nx;
        Dl t = pgf_face_tl((size_t)j * nx + ia, (size_t)j * nx + ib, k, st, nz, rdx,
                           p, pd, al, ald, alt, altd, php, phpd, mup, mupd,
                           pb, phb, mub, c1h, c2h, rdnw, fnm, fnp,
                           cf1, cf2, cf3, top_lid, cfn, cfn1);
        size_t out = R3(k, j, i, ny, nxf);
        if (nl) ru_t[out] = ru_t[out] - t.v;
        if (WRFPLUS_TL) ru_d[out] = ru_d[out] - t.d;
    }
    if (i < nx && ((!bnd_y && j <= ny) || (bnd_y && j > 0 && j < ny))) {
        int ja = j % ny, jb = (j - 1 + ny) % ny;
        Dl t = pgf_face_tl((size_t)ja * nx + i, (size_t)jb * nx + i, k, st, nz, rdy,
                           p, pd, al, ald, alt, altd, php, phpd, mup, mupd,
                           pb, phb, mub, c1h, c2h, rdnw, fnm, fnp,
                           cf1, cf2, cf3, top_lid, cfn, cfn1);
        size_t out = R3(k, j, i, nyf, nx);
        if (nl) rv_t[out] = rv_t[out] - t.v;
        if (WRFPLUS_TL) rv_d[out] = rv_d[out] - t.d;
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
    size_t rem = tid - (size_t)k * lay;
    int j = (int)(rem / nxf);
    int i = (int)(rem - (size_t)j * nxf);
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
    if (WRFPLUS_TL) rw_d[ix] = rw_d[ix] + buoy.d;
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

// =============================================================== rhs_ph
//
// woof's slow_geopotential_open with supplied = 0 and add_vertical = 1:
// at w level k = 1..nz (cell (j, i)) the tendency rph_t
//   - loses fnm hi + fnp lo, the vertical advection by ww (k < nz),
//   - gains (C_f(mu) g) w / msft,
//   - loses, per pass (0: y, 1: x, 2: WRF's second x pickup on specified
//     axes for orders 3-4), coef (f1 d1 + f0 d0) (stencil 2) or
//     coef (((f1 + f0) / 12 or 60) raw) (stencils 4 and 6), with the face
//     mass fluxes f = C_f(mu_face) (u_k + u_k-1) msfu (cfn/cfn1 at the top),
//   - on open (not specified) edges, loses the outer upwind term
//     (rd C_f(mu)) (speed d(php)) with speed clipped to inflow.
// The top level starts from 0 instead of rph_t.

#define GEO_PASSIVE                                                         \
    const wreal* __restrict__ phb, const wreal* __restrict__ mub,           \
    const wreal* __restrict__ rdnw, const wreal* __restrict__ fnm,          \
    const wreal* __restrict__ fnp, const wreal* __restrict__ c1f,           \
    const wreal* __restrict__ c2f, const wreal* __restrict__ msft,          \
    const wreal* __restrict__ msfu, const wreal* __restrict__ msfv

#define GEO_GEOM wreal qrdx, wreal qrdy, wreal cfn, wreal cfn1, int order, \
    int bnd_x, int bnd_y, int specified, int has_msf, int nz, int ny, int nx

static __device__ __forceinline__ float geo_min0(float v) { return fminf(v, 0.0f); }
static __device__ __forceinline__ double geo_min0(double v) { return fmin(v, 0.0); }
static __device__ __forceinline__ float geo_max0(float v) { return fmaxf(v, 0.0f); }
static __device__ __forceinline__ double geo_max0(double v) { return fmax(v, 0.0); }

// What the primal applies on pass (0 y, 1 x, 2 the second x pickup) at
// position pos: 0 nothing, 1 the outer upwind term, 2/4/6 the stencil.
static __device__ int geo_kind(int pass, int pos, int length, int bnd, int specified,
                               int order)
{
    int axis = pass != 0;
    int sx4 = specified && axis && (order == 3 || order == 4);
    if (pass == 2) return (sx4 && (pos == 1 || pos == length - 2)) ? 2 : 0;
    if (bnd && (pos == 0 || pos == length - 1) && !sx4) return specified ? 0 : 1;
    if (specified && axis && order >= 5 && (pos == 2 || pos == length - 3)) return 0;
    if (order <= 2 || (bnd && !sx4 && (pos == 1 || pos == length - 2))) return 2;
    if (order <= 4 || (bnd && (pos == 2 || pos == length - 3))) return 4;
    return 6;
}

// x face Q of row j: the u face read, the cells averaged for its mass.
struct GeoFace { size_t ca; size_t cb; int fidx; };

static __device__ GeoFace geo_xface(int face, int j, int nx, int phys)
{
    int fidx = face % nx;
    GeoFace F;
    F.ca = (size_t)j * nx + fidx;
    F.cb = (size_t)j * nx + (fidx - 1 + nx) % nx;
    F.fidx = (phys && face == nx) ? nx : fidx;
    return F;
}

static __device__ GeoFace geo_yface(int row, int i, int ny, int nx)
{
    int ridx = row % ny;
    GeoFace F;
    F.ca = (size_t)ridx * nx + i;
    F.cb = (size_t)((ridx - 1 + ny) % ny) * nx + i;
    F.fidx = ridx;
    return F;
}

// Face mass flux (fcx_value / fcy_value, strict).
static __device__ Dl geo_flux_tl(GeoFace F, int axis, int line, int k, int top, int ny, int nx,
                                 const wreal* uv, const wreal* uvd, const wreal* mup,
                                 const wreal* mupd, const wreal* mub, const wreal* c1f,
                                 const wreal* c2f, const wreal* msfuv, int has_msf, wreal cfn,
                                 wreal cfn1)
{
    Dl avg = (wreal)0.5 * (((ld(mup, mupd, F.ca) + ld(mup, mupd, F.cb)) + mub[F.ca]) + mub[F.cb]);
    Dl mass = c1f[k] * avg + c2f[k];
    size_t f, fo, at;
    if (axis) {
        f = R3(k - 1, line, F.fidx, ny, nx + 1);
        fo = top ? R3(k - 2, line, F.fidx, ny, nx + 1) : R3(k, line, F.fidx, ny, nx + 1);
        at = (size_t)line * (nx + 1) + F.fidx;
    } else {
        f = R3(k - 1, F.fidx, line, ny + 1, nx);
        fo = top ? R3(k - 2, F.fidx, line, ny + 1, nx) : R3(k, F.fidx, line, ny + 1, nx);
        at = (size_t)F.fidx * nx + line;
    }
    Dl wind = top ? cfn * ld(uv, uvd, f) + cfn1 * ld(uv, uvd, fo)
                  : ld(uv, uvd, fo) + ld(uv, uvd, f);
    Dl value = mass * wind;
    if (has_msf) value = value * msfuv[at];
    return value;
}

static __device__ __forceinline__ size_t geo_cell(int axis, int j, int i, int pos, int nx)
{
    return axis ? (size_t)j * nx + pos : (size_t)pos * nx + i;
}

// php difference at offset o about pos (wrf_phi_field_difference).
static __device__ Dl geo_pdiff_tl(const wreal* php, const wreal* phpd, int axis, int j, int i,
                                  int pos, int length, int o, int k, size_t st, int nx)
{
    size_t cp = geo_cell(axis, j, i, (pos + o + length) % length, nx);
    size_t cm = geo_cell(axis, j, i, (pos - o + length) % length, nx);
    return ld(php, phpd, (size_t)k * st + cp) - ld(php, phpd, (size_t)k * st + cm);
}

static __device__ wreal geo_bdiff(const wreal* phb, int axis, int j, int i, int pos, int length,
                                  int o, int k, size_t st, int nx)
{
    size_t cp = geo_cell(axis, j, i, (pos + o + length) % length, nx);
    size_t cm = geo_cell(axis, j, i, (pos - o + length) % length, nx);
    return phb[(size_t)k * st + cp] - phb[(size_t)k * st + cm];
}

// One pass's stencil, split into the terms its adjoint needs:
// contribution = coef (f1 d1 + f0 d0) (stencil 2) or coef (wf raw),
// wf = (f1 + f0) recip.
struct GeoStencil { Dl f0; Dl f1; Dl d0; Dl d1; Dl raw; wreal coef; wreal recip; };

static __device__ GeoStencil geo_stencil_tl(int pass, int kind, int k, int top, int j, int i,
                                            size_t st, const wreal* u, const wreal* ud,
                                            const wreal* v, const wreal* vd, const wreal* php,
                                            const wreal* phpd, const wreal* mup,
                                            const wreal* mupd, GEO_PASSIVE, GEO_GEOM)
{
    int axis = pass != 0;
    int pos = axis ? i : j, length = axis ? nx : ny;
    int sx4 = specified && axis && (order == 3 || order == 4);
    GeoStencil S;
    wreal rd = axis ? qrdx : qrdy;
    if (top) rd = (wreal)2 * rd;
    size_t c = (size_t)j * nx + i;
    S.coef = has_msf ? wdiv(rd, msft[c]) : rd;
    if (axis) {
        S.f0 = geo_flux_tl(geo_xface(i, j, nx, sx4), 1, j, k, top, ny, nx, u, ud, mup, mupd, mub,
                           c1f, c2f, msfu, has_msf, cfn, cfn1);
        S.f1 = geo_flux_tl(geo_xface(i + 1, j, nx, sx4), 1, j, k, top, ny, nx, u, ud, mup, mupd,
                           mub, c1f, c2f, msfu, has_msf, cfn, cfn1);
    } else {
        S.f0 = geo_flux_tl(geo_yface(j, i, ny, nx), 0, i, k, top, ny, nx, v, vd, mup, mupd, mub,
                           c1f, c2f, msfv, has_msf, cfn, cfn1);
        S.f1 = geo_flux_tl(geo_yface(j + 1, i, ny, nx), 0, i, k, top, ny, nx, v, vd, mup, mupd,
                           mub, c1f, c2f, msfv, has_msf, cfn, cfn1);
    }
    Dl zero = dl((wreal)0, (wreal)0);
    S.d0 = zero;
    S.d1 = zero;
    S.raw = zero;
    S.recip = (wreal)0;
    size_t ix = (size_t)k * st + c;
    if (kind == 2) {
        size_t cm = geo_cell(axis, j, i, (pos - 1 + length) % length, nx);
        size_t cp = geo_cell(axis, j, i, (pos + 1) % length, nx);
        size_t km = (size_t)k * st + cm, kp = (size_t)k * st + cp;
        S.d0 = (dl(phb[ix] - phb[km], (wreal)0) + ld(php, phpd, ix)) - ld(php, phpd, km);
        S.d1 = (dl(phb[kp] - phb[ix], (wreal)0) + ld(php, phpd, kp)) - ld(php, phpd, ix);
    } else {
        Dl p1 = geo_pdiff_tl(php, phpd, axis, j, i, pos, length, 1, k, st, nx);
        Dl p2 = geo_pdiff_tl(php, phpd, axis, j, i, pos, length, 2, k, st, nx);
        wreal b1 = geo_bdiff(phb, axis, j, i, pos, length, 1, k, st, nx);
        wreal b2 = geo_bdiff(phb, axis, j, i, pos, length, 2, k, st, nx);
        if (kind == 4) {
            S.raw = (((wreal)8 * p1 - p2) + (wreal)8 * b1) - b2;
            S.recip = wdiv((wreal)1, (wreal)12);
        } else {
            Dl p3 = geo_pdiff_tl(php, phpd, axis, j, i, pos, length, 3, k, st, nx);
            wreal b3 = geo_bdiff(phb, axis, j, i, pos, length, 3, k, st, nx);
            Dl raw = ((wreal)45 * p1 - (wreal)9 * p2) + p3;
            S.raw = ((raw + (wreal)45 * b1) - (wreal)9 * b2) + b3;
            S.recip = wdiv((wreal)1, (wreal)60);
        }
    }
    return S;
}

static __device__ __forceinline__ Dl geo_contribution(GeoStencil S, int kind)
{
    if (kind == 2) return S.coef * (S.f1 * S.d1 + S.f0 * S.d0);
    return S.coef * (((S.f1 + S.f0) * S.recip) * S.raw);
}

// The outer upwind term's parts: term = (rd mass) (speed diff).
struct GeoOuter { Dl speed; Dl diff; wreal rd; int pass_d; };

static __device__ GeoOuter geo_outer_tl(int axis, int k, int top, int j, int i, size_t st,
                                        const wreal* u, const wreal* ud, const wreal* v,
                                        const wreal* vd, const wreal* php, const wreal* phpd,
                                        const wreal* fnm, const wreal* fnp, wreal qrdx,
                                        wreal qrdy, int nz, int ny, int nx)
{
    int pos = axis ? i : j, length = axis ? nx : ny;
    int west_top = axis && pos == 0 && top;
    int kz = west_top ? k : (k < nz - 1 ? k : nz - 1);
    wreal hm = west_top ? (wreal)0 : fnm[kz], hp = west_top ? (wreal)0 : fnp[kz];
    Dl zero = dl((wreal)0, (wreal)0);
    Dl wl, wh;
    if (axis) {
        size_t face = (size_t)j * (nx + 1) + i, plane = (size_t)ny * (nx + 1);
        wh = west_top ? zero
                      : ld(u, ud, (size_t)kz * plane + face + 1) + ld(u, ud, (size_t)kz * plane + face);
        wl = ld(u, ud, (size_t)(kz - 1) * plane + face + 1) + ld(u, ud, (size_t)(kz - 1) * plane + face);
    } else {
        size_t face = (size_t)j * nx + i, plane = (size_t)(ny + 1) * nx;
        wh = ld(v, vd, (size_t)kz * plane + face + nx) + ld(v, vd, (size_t)kz * plane + face);
        wl = ld(v, vd, (size_t)(kz - 1) * plane + face + nx) + ld(v, vd, (size_t)(kz - 1) * plane + face);
    }
    Dl s = (wreal)0.5 * (hm * wh + hp * wl);
    GeoOuter O;
    // WRFPLUS's derivative of min(s, 0) is 0 at s = 0, of max(s, 0) is 1.
    if (pos == 0) {
        O.pass_d = s.v < (wreal)0;
        O.speed = dl(geo_min0(s.v), O.pass_d ? s.d : (wreal)0);
    } else {
        O.pass_d = s.v >= (wreal)0;
        O.speed = dl(geo_max0(s.v), O.pass_d ? s.d : (wreal)0);
    }
    size_t c = (size_t)j * nx + i, ix = (size_t)k * st + c;
    size_t ci = axis ? (pos == 0 ? c + 1 : c - 1) : (pos == 0 ? c + nx : c - nx);
    O.diff = pos == 0 ? ld(php, phpd, (size_t)k * st + ci) - ld(php, phpd, ix)
                      : ld(php, phpd, ix) - ld(php, phpd, (size_t)k * st + ci);
    O.rd = (wreal)4 * (axis ? qrdx : qrdy);
    (void)length;
    return O;
}

extern "C" __global__
void geo_tl(wreal* __restrict__ rph_t, wreal* __restrict__ rph_d,
            const wreal* __restrict__ ww, const wreal* __restrict__ wwd,
            const wreal* __restrict__ w, const wreal* __restrict__ wd,
            const wreal* __restrict__ u, const wreal* __restrict__ ud,
            const wreal* __restrict__ v, const wreal* __restrict__ vd,
            const wreal* __restrict__ php, const wreal* __restrict__ phpd,
            const wreal* __restrict__ mup, const wreal* __restrict__ mupd,
            GEO_PASSIVE, int nl, GEO_GEOM)
{
    size_t st = (size_t)ny * nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)nz * st) return;
    int k = (int)(tid / st) + 1;
    size_t c = tid - (size_t)(k - 1) * st, ix = (size_t)k * st + c;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx), top = k == nz;
    Dl tend = top ? dl((wreal)0, (wreal)0) : ld(rph_t, rph_d, ix);
    Dl mass = c1f[k] * (dl(mub[c], (wreal)0) + ld(mup, mupd, c)) + c2f[k];
    if (!top) {
        Dl dlo = ((ld(php, phpd, ix) - ld(php, phpd, ix - st)) + phb[ix]) - phb[ix - st];
        Dl dhi = ((ld(php, phpd, ix + st) - ld(php, phpd, ix)) + phb[ix + st]) - phb[ix];
        Dl lo = (((wreal)0.5 * (ld(ww, wwd, ix) + ld(ww, wwd, ix - st))) * rdnw[k - 1]) * dlo;
        Dl hi = (((wreal)0.5 * (ld(ww, wwd, ix + st) + ld(ww, wwd, ix))) * rdnw[k]) * dhi;
        tend = tend - (fnm[k] * hi + fnp[k] * lo);
    }
    Dl gw = (mass * BS_G) * ld(w, wd, ix);
    tend = tend + (has_msf ? gw / msft[c] : gw);
    for (int pass = 0; pass < 3; ++pass) {
        int axis = pass != 0;
        int kind = geo_kind(pass, axis ? i : j, axis ? nx : ny, axis ? bnd_x : bnd_y, specified,
                            order);
        if (kind < 2) continue;
        GeoStencil S = geo_stencil_tl(pass, kind, k, top, j, i, st, u, ud, v, vd, php, phpd, mup,
                                      mupd, phb, mub, rdnw, fnm, fnp, c1f, c2f, msft, msfu,
                                      msfv, qrdx, qrdy, cfn, cfn1, order, bnd_x, bnd_y,
                                      specified, has_msf, nz, ny, nx);
        tend = tend - geo_contribution(S, kind);
    }
    for (int axis = 0; axis < 2; ++axis) {
        if (geo_kind(axis, axis ? i : j, axis ? nx : ny, axis ? bnd_x : bnd_y, specified,
                     order) != 1)
            continue;
        GeoOuter O = geo_outer_tl(axis, k, top, j, i, st, u, ud, v, vd, php, phpd, fnm, fnp,
                                  qrdx, qrdy, nz, ny, nx);
        tend = tend - (O.rd * mass) * (O.speed * O.diff);
    }
    if (nl) rph_t[ix] = tend.v;
    if (WRFPLUS_TL) rph_d[ix] = tend.d;
}

// ---- adjoint
//
// geo_ad_cell writes, per w level k = 1..nz and cell, the local adjoints
// of each pass into scratch (every word):
//   sy (4 slots): a_f0, a_f1, s0, s1 for pass 0 (y);
//   sx (6 slots): a_f0, a_f1, s0, s1 for pass 1 (x), plus pass 2's face
//                 adjoints in a_f0/a_f1 and its s0, s1 in slots 4, 5.
// s0, s1 are the adjoints of d0, d1 (stencil 2) or s0 that of raw
// (stencils 4, 6).  On an outer-upwind cell (kind 1) the pass's stencil
// is absent and the slots hold a_f0 = the speed's (pre-clip) adjoint,
// a_f1 = the cell mass's, and s0 or s1 = the php difference's in the
// stencil-2 layout.  The gathers (php/ww/w levels, u faces, v faces,
// columns) then sum them; geo_ad_top zeroes the overwritten top level.

#define GEO_AD_IN                                                           \
    const wreal* __restrict__ ww, const wreal* __restrict__ w,              \
    const wreal* __restrict__ u, const wreal* __restrict__ v,               \
    const wreal* __restrict__ php, const wreal* __restrict__ mup

extern "C" __global__
void geo_ad_cell(const wreal* __restrict__ rph_a, GEO_AD_IN, wreal* __restrict__ sy,
                 wreal* __restrict__ sx, GEO_PASSIVE, GEO_GEOM)
{
    size_t st = (size_t)ny * nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)nz * st;
    if (tid >= n) return;
    int k = (int)(tid / st) + 1;
    size_t c = tid - (size_t)(k - 1) * st, ix = (size_t)k * st + c;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx), top = k == nz;
    wreal a = rph_a[ix];
    wreal out[10];
    for (int s = 0; s < 10; ++s) out[s] = (wreal)0;
    for (int pass = 0; pass < 3; ++pass) {
        int axis = pass != 0;
        int kind = geo_kind(pass, axis ? i : j, axis ? nx : ny, axis ? bnd_x : bnd_y, specified,
                            order);
        int base = axis ? 4 : 0;
        if (kind == 1) {
            // outer upwind term: tend -= (rd mass) (speed diff)
            wreal mass = c1f[k] * (mub[c] + mup[c]) + c2f[k];
            GeoOuter O = geo_outer_tl(axis, k, top, j, i, st, u, u, v, v, php, php, fnm, fnp,
                                      qrdx, qrdy, nz, ny, nx);
            wreal g = -a;
            out[base] = O.pass_d ? g * (O.rd * mass) * O.diff.v : (wreal)0;
            out[base + 1] = g * O.rd * (O.speed.v * O.diff.v);
            wreal adiff = g * (O.rd * mass) * O.speed.v;
            if (axis ? i == 0 : j == 0) out[base + 3] = adiff;
            else out[base + 2] = adiff;
            continue;
        }
        if (kind == 0) continue;
        GeoStencil S = geo_stencil_tl(pass, kind, k, top, j, i, st, u, u, v, v, php, php, mup,
                                      mup, phb, mub, rdnw, fnm, fnp, c1f, c2f, msft, msfu, msfv,
                                      qrdx, qrdy, cfn, cfn1, order, bnd_x, bnd_y, specified,
                                      has_msf, nz, ny, nx);
        wreal g = -a * S.coef;
        wreal af0, af1, s0, s1;
        if (kind == 2) {
            af0 = g * S.d0.v;
            af1 = g * S.d1.v;
            s0 = g * S.f0.v;
            s1 = g * S.f1.v;
        } else {
            af0 = g * S.raw.v * S.recip;
            af1 = af0;
            s0 = g * ((S.f1.v + S.f0.v) * S.recip);
            s1 = (wreal)0;
        }
        out[base] += af0;
        out[base + 1] += af1;
        if (pass == 2) {
            out[8] = s0;
            out[9] = s1;
        } else {
            out[base + 2] = s0;
            out[base + 3] = s1;
        }
    }
    for (int s = 0; s < 4; ++s) sy[(size_t)s * n + tid] = out[s];
    for (int s = 0; s < 6; ++s) sx[(size_t)s * n + tid] = out[4 + s];
}

#define GEO_GATHER_ARGS                                                     \
    const wreal* __restrict__ rph_a, GEO_AD_IN, const wreal* __restrict__ sy, \
    const wreal* __restrict__ sx, wreal* __restrict__ aww,                  \
    wreal* __restrict__ aw, wreal* __restrict__ au, wreal* __restrict__ av, \
    wreal* __restrict__ aphp, wreal* __restrict__ amup, GEO_PASSIVE, GEO_GEOM

// Scratch slot s at w level k (1..nz) and cell c.
static __device__ __forceinline__ wreal geo_slot(const wreal* sc, int s, int k, size_t c,
                                                 size_t st, int nz)
{
    return sc[(size_t)s * nz * st + (size_t)(k - 1) * st + c];
}

// Weight of php at offset t from a pass of the given kind (stencil 2
// holds d0 = php[0] - php[-1], d1 = php[1] - php[0]).
static __device__ wreal geo_pweight(int kind, int t, wreal s0, wreal s1)
{
    if (kind == 1 || kind == 2) {
        if (t == -1) return -s0;
        if (t == 0) return s0 - s1;
        if (t == 1) return s1;
        return (wreal)0;
    }
    int at = t < 0 ? -t : t;
    wreal sg = t < 0 ? (wreal)-1 : (wreal)1;
    if (kind == 4) {
        if (at == 1) return sg * (wreal)8 * s0;
        if (at == 2) return -sg * s0;
        return (wreal)0;
    }
    if (at == 1) return sg * (wreal)45 * s0;
    if (at == 2) return -sg * (wreal)9 * s0;
    if (at == 3) return sg * s0;
    return (wreal)0;
}

static __device__ __forceinline__ wreal geo_seed(const wreal* rph_a, int k, size_t c,
                                                 size_t st, int nz)
{
    return (k >= 1 && k <= nz) ? rph_a[(size_t)k * st + c] : (wreal)0;
}

// php, ww and w at w level m = 0..nz.
extern "C" __global__ void geo_ad_lev(GEO_GATHER_ARGS)
{
    size_t st = (size_t)ny * nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)(nz + 1) * st) return;
    int m = (int)(tid / st);
    size_t c = tid - (size_t)m * st;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    wreal gphp = 0, gww = 0;
    // vertical advection at levels k = m - 1, m, m + 1 (k in 1..nz-1)
    for (int k = m - 1; k <= m + 1; ++k) {
        if (k < 1 || k > nz - 1) continue;
        size_t ix = (size_t)k * st + c;
        wreal a = rph_a[ix];
        wreal ahi = -a * fnm[k], alo = -a * fnp[k];
        wreal dlo = ((php[ix] - php[ix - st]) + phb[ix]) - phb[ix - st];
        wreal dhi = ((php[ix + st] - php[ix]) + phb[ix + st]) - phb[ix];
        wreal whi = ((wreal)0.5 * (ww[ix + st] + ww[ix])) * rdnw[k];
        wreal wlo = ((wreal)0.5 * (ww[ix] + ww[ix - st])) * rdnw[k - 1];
        wreal adh = ahi * whi, adl = alo * wlo;
        wreal awhi = (wreal)0.5 * rdnw[k] * ahi * dhi;
        wreal awlo = (wreal)0.5 * rdnw[k - 1] * alo * dlo;
        if (k == m) {
            gphp += adl - adh;
            gww += awhi + awlo;
        } else if (k == m - 1) {
            gphp += adh;
            gww += awhi;
        } else {
            gphp -= adl;
            gww += awlo;
        }
    }
    if (m >= 1) {
        size_t n = (size_t)nz * st;
        // horizontal stencils and outer terms of the cells reading php(m)
        for (int pass = 0; pass < 3; ++pass) {
            int axis = pass != 0;
            int pos = axis ? i : j, length = axis ? nx : ny, bnd = axis ? bnd_x : bnd_y;
            const wreal* sc = axis ? sx : sy;
            int s0i = pass == 2 ? 4 : 2, s1i = pass == 2 ? 5 : 3;
            for (int t = -3; t <= 3; ++t) {
                int r = ((pos - t) % length + 4 * length) % length;
                int kind = geo_kind(pass, r, length, bnd, specified, order);
                if (kind == 0) continue;
                size_t cr = geo_cell(axis, j, i, r, nx);
                size_t at = (size_t)(m - 1) * st + cr;
                gphp += geo_pweight(kind, t, sc[(size_t)s0i * n + at], sc[(size_t)s1i * n + at]);
            }
        }
        // g w
        size_t ix = (size_t)m * st + c;
        wreal a = rph_a[ix];
        wreal mass = c1f[m] * (mub[c] + mup[c]) + c2f[m];
        wreal s = has_msf ? a / msft[c] : a;
        aw[ix] += s * mass * BS_G;
    }
    aphp[tid] += gphp;
    aww[tid] += gww;
}

// Adjoint of the face flux value at x face Q (0..nx), w level k: the
// stencil cells that read it (their pass 1/2 slots; not outer cells).
static __device__ wreal geo_aflux_x(const wreal* sx, int k, int j, int Q, size_t st, GEO_GEOM)
{
    int phys = specified && (order == 3 || order == 4);
    wreal g = 0;
    int i0 = -1, i1 = -1;
    if (Q < nx) i0 = Q;
    if (Q >= 1 && (Q < nx || phys)) i1 = Q - 1;
    else if (Q == 0 && !phys) i1 = nx - 1;
    if (i0 >= 0 && geo_kind(1, i0, nx, bnd_x, specified, order) != 1)
        g += geo_slot(sx, 0, k, (size_t)j * nx + i0, st, nz);
    if (i1 >= 0 && geo_kind(1, i1, nx, bnd_x, specified, order) != 1)
        g += geo_slot(sx, 1, k, (size_t)j * nx + i1, st, nz);
    return g;
}

static __device__ wreal geo_aflux_y(const wreal* sy, int k, int R, int i, size_t st, GEO_GEOM)
{
    wreal g = 0;
    int j0 = -1, j1 = -1;
    if (R < ny) j0 = R;
    if (R >= 1 && R < ny) j1 = R - 1;
    else if (R == 0) j1 = ny - 1;
    if (j0 >= 0 && geo_kind(0, j0, ny, bnd_y, specified, order) != 1)
        g += geo_slot(sy, 0, k, (size_t)j0 * nx + i, st, nz);
    if (j1 >= 0 && geo_kind(0, j1, ny, bnd_y, specified, order) != 1)
        g += geo_slot(sy, 1, k, (size_t)j1 * nx + i, st, nz);
    return g;
}

// Face mass C_f(avg mu) at face F, w level k.
static __device__ __forceinline__ wreal geo_fmass(GeoFace F, int k, const wreal* mup,
                                                  const wreal* mub, const wreal* c1f,
                                                  const wreal* c2f)
{
    wreal avg = (wreal)0.5 * (((mup[F.ca] + mup[F.cb]) + mub[F.ca]) + mub[F.cb]);
    return c1f[k] * avg + c2f[k];
}

// Weight of a wind word at mass level kk in the face flux at w level k.
static __device__ __forceinline__ wreal geo_wind_w(int k, int kk, int nz, wreal cfn, wreal cfn1)
{
    if (k == nz) return kk == nz - 1 ? cfn : (kk == nz - 2 ? cfn1 : (wreal)0);
    return (kk == k || kk == k - 1) ? (wreal)1 : (wreal)0;
}

// The outer term's speed adjoint reaching wind level kk, faces of the
// edge cell e on the axis, w level k (kz as the primal picks it).
static __device__ wreal geo_outer_wind(const wreal* sc, int axis, int e, int line, int kk,
                                       size_t st, const wreal* fnm, const wreal* fnp, int nz,
                                       int nx)
{
    wreal g = 0;
    for (int k = 1; k <= nz; ++k) {
        int west_top = axis && e == 0 && k == nz;
        if (west_top) continue;
        int kz = k < nz - 1 ? k : nz - 1;
        size_t c = axis ? (size_t)line * nx + e : (size_t)e * nx + line;
        wreal as = geo_slot(sc, 0, k, c, st, nz);
        if (kk == kz) g += (wreal)0.5 * fnm[kz] * as;
        if (kk == kz - 1) g += (wreal)0.5 * fnp[kz] * as;
    }
    return g;
}

// u at mass level kk, face (j, Q), Q = 0..nx.
extern "C" __global__ void geo_ad_u(GEO_GATHER_ARGS)
{
    size_t st = (size_t)ny * nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane = (size_t)ny * (nx + 1);
    if (tid >= (size_t)nz * plane) return;
    int kk = (int)(tid / plane);
    size_t rem = tid - (size_t)kk * plane;
    int j = (int)(rem / (nx + 1)), Q = (int)(rem - (size_t)j * (nx + 1));
    int phys = specified && (order == 3 || order == 4);
    wreal g = 0;
    if (Q < nx || phys) {
        GeoFace F = geo_xface(Q, j, nx, phys);
        wreal msf = has_msf ? msfu[(size_t)j * (nx + 1) + Q] : (wreal)1;
        int ks[3] = {kk, kk + 1, nz};
        for (int t = 0; t < 3; ++t) {
            int k = ks[t];
            if (k < 1 || k > nz) continue;
            if (t == 2 && (k == kk || k == kk + 1)) continue;
            wreal wgt = geo_wind_w(k, kk, nz, cfn, cfn1);
            if (wgt == (wreal)0) continue;
            wreal af = geo_aflux_x(sx, k, j, Q, st, qrdx, qrdy, cfn, cfn1, order, bnd_x, bnd_y,
                                   specified, has_msf, nz, ny, nx);
            g += af * msf * geo_fmass(F, k, mup, mub, c1f, c2f) * wgt;
        }
    }
    if (bnd_x && !specified) {
        if (Q <= 1) g += geo_outer_wind(sx, 1, 0, j, kk, st, fnm, fnp, nz, nx);
        if (Q >= nx - 1) g += geo_outer_wind(sx, 1, nx - 1, j, kk, st, fnm, fnp, nz, nx);
    }
    au[tid] += g;
}

// v at mass level kk, face (R, i), R = 0..ny.
extern "C" __global__ void geo_ad_v(GEO_GATHER_ARGS)
{
    size_t st = (size_t)ny * nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane = (size_t)(ny + 1) * nx;
    if (tid >= (size_t)nz * plane) return;
    int kk = (int)(tid / plane);
    size_t rem = tid - (size_t)kk * plane;
    int R = (int)(rem / nx), i = (int)(rem - (size_t)R * nx);
    wreal g = 0;
    if (R < ny) {
        GeoFace F = geo_yface(R, i, ny, nx);
        wreal msf = has_msf ? msfv[(size_t)R * nx + i] : (wreal)1;
        int ks[3] = {kk, kk + 1, nz};
        for (int t = 0; t < 3; ++t) {
            int k = ks[t];
            if (k < 1 || k > nz) continue;
            if (t == 2 && (k == kk || k == kk + 1)) continue;
            wreal wgt = geo_wind_w(k, kk, nz, cfn, cfn1);
            if (wgt == (wreal)0) continue;
            wreal af = geo_aflux_y(sy, k, R, i, st, qrdx, qrdy, cfn, cfn1, order, bnd_x, bnd_y,
                                   specified, has_msf, nz, ny, nx);
            g += af * msf * geo_fmass(F, k, mup, mub, c1f, c2f) * wgt;
        }
    }
    if (bnd_y && !specified) {
        if (R <= 1) g += geo_outer_wind(sy, 0, 0, i, kk, st, fnm, fnp, nz, nx);
        if (R >= ny - 1) g += geo_outer_wind(sy, 0, ny - 1, i, kk, st, fnm, fnp, nz, nx);
    }
    av[tid] += g;
}

// Sum over w levels of the face flux's mass adjoint (times 1/2 for each
// cell of the average): c1f a_f wind msf.
static __device__ wreal geo_face_amass(int axis, int Q, int line, size_t st, const wreal* sc,
                                       const wreal* uv, GEO_GEOM, const wreal* c1f,
                                       const wreal* msfuv)
{
    int phys = axis && specified && (order == 3 || order == 4);
    int fidx = axis ? ((phys && Q == nx) ? nx : Q % nx) : Q % ny;
    wreal msf = (wreal)1;
    if (has_msf) msf = axis ? msfuv[(size_t)line * (nx + 1) + fidx] : msfuv[(size_t)fidx * nx + line];
    wreal g = 0;
    for (int k = 1; k <= nz; ++k) {
        wreal af = axis ? geo_aflux_x(sc, k, line, Q, st, qrdx, qrdy, cfn, cfn1, order, bnd_x,
                                      bnd_y, specified, has_msf, nz, ny, nx)
                        : geo_aflux_y(sc, k, Q, line, st, qrdx, qrdy, cfn, cfn1, order, bnd_x,
                                      bnd_y, specified, has_msf, nz, ny, nx);
        if (af == (wreal)0) continue;
        size_t f, fo;
        if (axis) {
            f = R3(k - 1, line, fidx, ny, nx + 1);
            fo = k == nz ? R3(k - 2, line, fidx, ny, nx + 1) : R3(k, line, fidx, ny, nx + 1);
        } else {
            f = R3(k - 1, fidx, line, ny + 1, nx);
            fo = k == nz ? R3(k - 2, fidx, line, ny + 1, nx) : R3(k, fidx, line, ny + 1, nx);
        }
        wreal wind = k == nz ? cfn * uv[f] + cfn1 * uv[fo] : uv[fo] + uv[f];
        g += c1f[k] * af * wind * msf;
    }
    return (wreal)0.5 * g;
}

// mup at column (j, i).
extern "C" __global__ void geo_ad_mup(GEO_GATHER_ARGS)
{
    size_t st = (size_t)ny * nx;
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= st) return;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    int kx = geo_kind(1, i, nx, bnd_x, specified, order);
    int ky = geo_kind(0, j, ny, bnd_y, specified, order);
    wreal g = 0;
    for (int k = 1; k <= nz; ++k) {
        size_t ix = (size_t)k * st + c;
        wreal a = rph_a[ix];
        wreal s = has_msf ? a / msft[c] : a;
        wreal amass = s * w[ix] * BS_G;
        if (kx == 1) amass += geo_slot(sx, 1, k, c, st, nz);
        if (ky == 1) amass += geo_slot(sy, 1, k, c, st, nz);
        g += c1f[k] * amass;
    }
    int phys = specified && (order == 3 || order == 4);
    // x faces whose mass averages this cell: as ca (Q = i; Q = nx when
    // physical and i = 0) and as cb (Q = i + 1 mod nx; Q = nx when
    // physical and i = nx - 1).
    g += geo_face_amass(1, i, j, st, sx, u, qrdx, qrdy, cfn, cfn1, order, bnd_x, bnd_y,
                        specified, has_msf, nz, ny, nx, c1f, msfu);
    g += geo_face_amass(1, (i + 1) % nx, j, st, sx, u, qrdx, qrdy, cfn, cfn1, order, bnd_x,
                        bnd_y, specified, has_msf, nz, ny, nx, c1f, msfu);
    if (phys && (i == 0 || i == nx - 1))
        g += geo_face_amass(1, nx, j, st, sx, u, qrdx, qrdy, cfn, cfn1, order, bnd_x, bnd_y,
                            specified, has_msf, nz, ny, nx, c1f, msfu)
             * (wreal)((i == 0) + (i == nx - 1));
    g += geo_face_amass(0, j, i, st, sy, v, qrdx, qrdy, cfn, cfn1, order, bnd_x, bnd_y,
                        specified, has_msf, nz, ny, nx, c1f, msfv);
    g += geo_face_amass(0, (j + 1) % ny, i, st, sy, v, qrdx, qrdy, cfn, cfn1, order, bnd_x,
                        bnd_y, specified, has_msf, nz, ny, nx, c1f, msfv);
    amup[c] += g;
}

// The top level's tendency is overwritten: its adjoint is consumed.
extern "C" __global__ void geo_ad_top(wreal* __restrict__ rph_a, int nz, int ny, int nx)
{
    size_t st = (size_t)ny * nx;
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= st) return;
    rph_a[(size_t)nz * st + c] = (wreal)0;
}

// ================================================== coriolis + curvature
//
// woof's strict coriolis_curvature (woof/core/kernels/coriolis_map.cu,
// the GPUWM_WRF_EXACT_C_BIGSTEP path: coriolis then curvature, each its
// own add/subtract): ru_t, rv_t and rw_t gain WRF's coriolis and
// curvature terms.  Active: the coupled momenta ru, rv, the winds u, v, w
// and the total column mass mut (rw = (c1f mut + c2f) w / msft is formed
// inline).  The boundary-normal faces of a bounded axis are left alone.

#define CC_PASSIVE                                                          \
    const wreal* __restrict__ msft, const wreal* __restrict__ msfu,         \
    const wreal* __restrict__ msfv, const wreal* __restrict__ f,            \
    const wreal* __restrict__ e, const wreal* __restrict__ sina,            \
    const wreal* __restrict__ cosa, const wreal* __restrict__ c1f,          \
    const wreal* __restrict__ c2f, const wreal* __restrict__ fzm,           \
    const wreal* __restrict__ fzp

#define CC_GEOM wreal rdx, wreal rdy, wreal rerad, int bnd_x, int bnd_y, int nz, int ny, int nx

static __device__ __forceinline__ Dl cc_rw(const wreal* w, const wreal* wd, const wreal* mut,
                                           const wreal* mutd, const wreal* msft,
                                           const wreal* c1f, const wreal* c2f, int kf,
                                           size_t c, size_t st)
{
    Dl mass = c1f[kf] * ld(mut, mutd, c) + c2f[kf];
    return (mass * ld(w, wd, (size_t)kf * st + c)) / msft[c];
}

// vxgm at mass point (j, i): u and v faces read directly (no wrap).
static __device__ __forceinline__ Dl cc_vxgm(const wreal* u, const wreal* ud, const wreal* v,
                                             const wreal* vd, const wreal* msfu,
                                             const wreal* msfv, wreal rdx, wreal rdy, int k,
                                             int j, int i, int ny, int nx)
{
    Dl ub = (wreal)0.5 * (ld(u, ud, R3(k, j, i, ny, nx + 1)) + ld(u, ud, R3(k, j, i + 1, ny, nx + 1)));
    Dl vb = (wreal)0.5 * (ld(v, vd, R3(k, j, i, ny + 1, nx)) + ld(v, vd, R3(k, j + 1, i, ny + 1, nx)));
    wreal dmv = msfv[(size_t)(j + 1) * nx + i] - msfv[(size_t)j * nx + i];
    wreal dmu = msfu[(size_t)j * (nx + 1) + i + 1] - msfu[(size_t)j * (nx + 1) + i];
    return (ub * dmv) * rdy - (vb * dmu) * rdx;
}

// 4-point rw at levels k, k+1 of cells cb (first) and ca, the primal's
// strict order.
static __device__ __forceinline__ Dl cc_rw4(const wreal* w, const wreal* wd, const wreal* mut,
                                            const wreal* mutd, const wreal* msft,
                                            const wreal* c1f, const wreal* c2f, int k,
                                            size_t ca, size_t cb, size_t st)
{
    return (wreal)0.25 * (((cc_rw(w, wd, mut, mutd, msft, c1f, c2f, k + 1, cb, st)
                            + cc_rw(w, wd, mut, mutd, msft, c1f, c2f, k, cb, st))
                           + cc_rw(w, wd, mut, mutd, msft, c1f, c2f, k + 1, ca, st))
                          + cc_rw(w, wd, mut, mutd, msft, c1f, c2f, k, ca, st));
}

// The half->full interpolation 0.5 (fzm (a + b)_k + fzp (a + b)_{k-1}).
static __device__ __forceinline__ Dl cc_full(const wreal* a, const wreal* ad, size_t p0,
                                             size_t p1, size_t lev, int k, const wreal* fzm,
                                             const wreal* fzp)
{
    Dl hi = fzm[k] * (ld(a, ad, p0 + (size_t)k * lev) + ld(a, ad, p1 + (size_t)k * lev));
    Dl lo = fzp[k] * (ld(a, ad, p0 + (size_t)(k - 1) * lev) + ld(a, ad, p1 + (size_t)(k - 1) * lev));
    return (wreal)0.5 * (hi + lo);
}

struct CCPos { int k, j, i; };

static __device__ __forceinline__ CCPos cc_pos(size_t tid, int ny, int nx)
{
    size_t plane = (size_t)(ny + 1) * (nx + 1);
    CCPos p;
    p.k = (int)(tid / plane);
    size_t r = tid - (size_t)p.k * plane;
    p.j = (int)(r / (nx + 1));
    p.i = (int)(r - (size_t)p.j * (nx + 1));
    return p;
}

extern "C" __global__
void cc_tl(wreal* __restrict__ ru_t, wreal* __restrict__ ru_td,
           wreal* __restrict__ rv_t, wreal* __restrict__ rv_td,
           wreal* __restrict__ rw_t, wreal* __restrict__ rw_td,
           const wreal* __restrict__ ru, const wreal* __restrict__ rud,
           const wreal* __restrict__ rv, const wreal* __restrict__ rvd,
           const wreal* __restrict__ u, const wreal* __restrict__ ud,
           const wreal* __restrict__ v, const wreal* __restrict__ vd,
           const wreal* __restrict__ w, const wreal* __restrict__ wd,
           const wreal* __restrict__ mut, const wreal* __restrict__ mutd,
           CC_PASSIVE, int nl, CC_GEOM)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)nz * (ny + 1) * (nx + 1)) return;
    CCPos q = cc_pos(tid, ny, nx);
    int k = q.k, j = q.j, i = q.i;
    size_t st = (size_t)ny * nx;
    if (j < ny && (!bnd_x || (i > 0 && i < nx))) {
        int ic = i % nx, imc = (i - 1 + nx) % nx;
        size_t ca = (size_t)j * nx + ic, cb = (size_t)j * nx + imc;
        Dl rv4 = (wreal)0.25 * (((ld(rv, rvd, R3(k, j + 1, imc, ny + 1, nx))
                                  + ld(rv, rvd, R3(k, j + 1, ic, ny + 1, nx)))
                                 + ld(rv, rvd, R3(k, j, imc, ny + 1, nx)))
                                + ld(rv, rvd, R3(k, j, ic, ny + 1, nx)));
        Dl rw4 = cc_rw4(w, wd, mut, mutd, msft, c1f, c2f, k, ca, cb, st);
        wreal fh = (wreal)0.5 * (f[ca] + f[cb]);
        wreal eh = (wreal)0.5 * (e[ca] + e[cb]);
        wreal ch = (wreal)0.5 * (cosa[ca] + cosa[cb]);
        Dl vx = (wreal)0.5 * (cc_vxgm(u, ud, v, vd, msfu, msfv, rdx, rdy, k, j, ic, ny, nx)
                              + cc_vxgm(u, ud, v, vd, msfu, msfv, rdx, rdy, k, j, imc, ny, nx));
        size_t o = R3(k, j, i, ny, nx + 1);
        Dl t = ld(ru_t, ru_td, o);
        t = (t + fh * rv4) - (eh * ch) * rw4;
        t = (t + vx * rv4) - (ld(u, ud, o) * rerad) * rw4;
        if (nl) ru_t[o] = t.v;
        if (WRFPLUS_TL) ru_td[o] = t.d;
    }
    if (i < nx && (!bnd_y || (j > 0 && j < ny))) {
        int jc = j % ny, jmc = (j - 1 + ny) % ny;
        size_t ca = (size_t)jc * nx + i, cb = (size_t)jmc * nx + i;
        Dl ru4 = (wreal)0.25 * (((ld(ru, rud, R3(k, jc, i, ny, nx + 1))
                                  + ld(ru, rud, R3(k, jc, i + 1, ny, nx + 1)))
                                 + ld(ru, rud, R3(k, jmc, i, ny, nx + 1)))
                                + ld(ru, rud, R3(k, jmc, i + 1, ny, nx + 1)));
        Dl rw4 = cc_rw4(w, wd, mut, mutd, msft, c1f, c2f, k, ca, cb, st);
        wreal fh = (wreal)0.5 * (f[ca] + f[cb]);
        wreal eh = (wreal)0.5 * (e[ca] + e[cb]);
        wreal sh = (wreal)0.5 * (sina[ca] + sina[cb]);
        Dl vx = (wreal)0.5 * (cc_vxgm(u, ud, v, vd, msfu, msfv, rdx, rdy, k, jc, i, ny, nx)
                              + cc_vxgm(u, ud, v, vd, msfu, msfv, rdx, rdy, k, jmc, i, ny, nx));
        size_t o = R3(k, j, i, ny + 1, nx);
        Dl t = ld(rv_t, rv_td, o);
        t = (t - fh * ru4) + (eh * sh) * rw4;
        t = (t - vx * ru4) - (ld(v, vd, o) * rerad) * rw4;
        if (nl) rv_t[o] = t.v;
        if (WRFPLUS_TL) rv_td[o] = t.d;
    }
    if (i < nx && j < ny && k >= 1) {
        size_t c = (size_t)j * nx + i;
        size_t ul = (size_t)ny * (nx + 1), vl = (size_t)(ny + 1) * nx;
        size_t u0 = (size_t)j * (nx + 1) + i, v0 = (size_t)j * nx + i;
        Dl ruf = cc_full(ru, rud, u0, u0 + 1, ul, k, fzm, fzp);
        Dl uf = cc_full(u, ud, u0, u0 + 1, ul, k, fzm, fzp);
        Dl rvf = cc_full(rv, rvd, v0, v0 + nx, vl, k, fzm, fzp);
        Dl vf = cc_full(v, vd, v0, v0 + nx, vl, k, fzm, fzp);
        size_t o = (size_t)k * st + c;
        Dl t = ld(rw_t, rw_td, o);
        t = t + e[c] * (cosa[c] * ruf - sina[c] * rvf);
        t = t + rerad * (ruf * uf + rvf * vf);
        if (nl) rw_t[o] = t.v;
        if (WRFPLUS_TL) rw_td[o] = t.d;
    }
}

// AD pass 1, one thread per (k, j, i) of the union grid: each output's
// adjoint times the partials of its local intermediates.
//   su (4, nz, ny, nx+1): a_rv4, a_rw4, a_vx, a_u (own face)
//   sv (4, nz, ny+1, nx): a_ru4, a_rw4, a_vx, a_v (own face)
//   sw (4, nz, ny, nx):   a_ruf, a_uf, a_rvf, a_vf (zero at k = 0)
// Every word is written (zero on the faces the primal skips).
extern "C" __global__
void cc_ad_face(const wreal* __restrict__ ru_a, const wreal* __restrict__ rv_a,
                const wreal* __restrict__ rw_a, const wreal* __restrict__ ru,
                const wreal* __restrict__ rv, const wreal* __restrict__ u,
                const wreal* __restrict__ v, const wreal* __restrict__ w,
                const wreal* __restrict__ mut, wreal* __restrict__ su,
                wreal* __restrict__ sv, wreal* __restrict__ sw, CC_PASSIVE, CC_GEOM)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)nz * (ny + 1) * (nx + 1)) return;
    CCPos q = cc_pos(tid, ny, nx);
    int k = q.k, j = q.j, i = q.i;
    size_t st = (size_t)ny * nx;
    wreal zero = (wreal)0;
    if (j < ny) {
        size_t o = R3(k, j, i, ny, nx + 1), n = (size_t)nz * ny * (nx + 1);
        wreal s0 = zero, s1 = zero, s2 = zero, s3 = zero;
        if (!bnd_x || (i > 0 && i < nx)) {
            int ic = i % nx, imc = (i - 1 + nx) % nx;
            size_t ca = (size_t)j * nx + ic, cb = (size_t)j * nx + imc;
            wreal rv4 = (wreal)0.25 * (((rv[R3(k, j + 1, imc, ny + 1, nx)]
                                         + rv[R3(k, j + 1, ic, ny + 1, nx)])
                                        + rv[R3(k, j, imc, ny + 1, nx)])
                                       + rv[R3(k, j, ic, ny + 1, nx)]);
            wreal rw4 = cc_rw4(w, w, mut, mut, msft, c1f, c2f, k, ca, cb, st).v;
            wreal fh = (wreal)0.5 * (f[ca] + f[cb]);
            wreal eh = (wreal)0.5 * (e[ca] + e[cb]);
            wreal ch = (wreal)0.5 * (cosa[ca] + cosa[cb]);
            wreal vx = (wreal)0.5 * (cc_vxgm(u, u, v, v, msfu, msfv, rdx, rdy, k, j, ic, ny, nx).v
                                     + cc_vxgm(u, u, v, v, msfu, msfv, rdx, rdy, k, j, imc, ny, nx).v);
            wreal a = ru_a[o];
            s0 = a * (fh + vx);
            s1 = -a * (eh * ch + u[o] * rerad);
            s2 = a * rv4;
            s3 = -a * rerad * rw4;
        }
        su[o] = s0;
        su[n + o] = s1;
        su[2 * n + o] = s2;
        su[3 * n + o] = s3;
    }
    if (i < nx) {
        size_t o = R3(k, j, i, ny + 1, nx), n = (size_t)nz * (ny + 1) * nx;
        wreal s0 = zero, s1 = zero, s2 = zero, s3 = zero;
        if (!bnd_y || (j > 0 && j < ny)) {
            int jc = j % ny, jmc = (j - 1 + ny) % ny;
            size_t ca = (size_t)jc * nx + i, cb = (size_t)jmc * nx + i;
            wreal ru4 = (wreal)0.25 * (((ru[R3(k, jc, i, ny, nx + 1)]
                                         + ru[R3(k, jc, i + 1, ny, nx + 1)])
                                        + ru[R3(k, jmc, i, ny, nx + 1)])
                                       + ru[R3(k, jmc, i + 1, ny, nx + 1)]);
            wreal rw4 = cc_rw4(w, w, mut, mut, msft, c1f, c2f, k, ca, cb, st).v;
            wreal fh = (wreal)0.5 * (f[ca] + f[cb]);
            wreal eh = (wreal)0.5 * (e[ca] + e[cb]);
            wreal sh = (wreal)0.5 * (sina[ca] + sina[cb]);
            wreal vx = (wreal)0.5 * (cc_vxgm(u, u, v, v, msfu, msfv, rdx, rdy, k, jc, i, ny, nx).v
                                     + cc_vxgm(u, u, v, v, msfu, msfv, rdx, rdy, k, jmc, i, ny, nx).v);
            wreal a = rv_a[o];
            s0 = -a * (fh + vx);
            s1 = a * (eh * sh - v[o] * rerad);
            s2 = -a * ru4;
            s3 = -a * rerad * rw4;
        }
        sv[o] = s0;
        sv[n + o] = s1;
        sv[2 * n + o] = s2;
        sv[3 * n + o] = s3;
    }
    if (i < nx && j < ny) {
        size_t c = (size_t)j * nx + i, o = (size_t)k * st + c, n = (size_t)nz * st;
        wreal s0 = zero, s1 = zero, s2 = zero, s3 = zero;
        if (k >= 1) {
            size_t ul = (size_t)ny * (nx + 1), vl = (size_t)(ny + 1) * nx;
            size_t u0 = (size_t)j * (nx + 1) + i, v0 = (size_t)j * nx + i;
            wreal ruf = cc_full(ru, ru, u0, u0 + 1, ul, k, fzm, fzp).v;
            wreal uf = cc_full(u, u, u0, u0 + 1, ul, k, fzm, fzp).v;
            wreal rvf = cc_full(rv, rv, v0, v0 + nx, vl, k, fzm, fzp).v;
            wreal vf = cc_full(v, v, v0, v0 + nx, vl, k, fzm, fzp).v;
            wreal a = rw_a[o];
            s0 = a * (e[c] * cosa[c] + rerad * uf);
            s1 = a * rerad * ruf;
            s2 = a * (rerad * vf - e[c] * sina[c]);
            s3 = a * rerad * rvf;
        }
        sw[o] = s0;
        sw[n + o] = s1;
        sw[2 * n + o] = s2;
        sw[3 * n + o] = s3;
    }
}

// a_vxgm at mass point (k, j, c): the u faces whose ic or imc is c and the
// v rows whose jc or jmc is j (each read counted once).  sg (nz, ny, nx).
extern "C" __global__
void cc_ad_vxgm(const wreal* __restrict__ su, const wreal* __restrict__ sv,
                wreal* __restrict__ sg, int nz, int ny, int nx)
{
    size_t st = (size_t)ny * nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)nz * st) return;
    int k = (int)(tid / st);
    int j = (int)((tid - (size_t)k * st) / nx);
    int c = (int)(tid - (size_t)k * st - (size_t)j * nx);
    const wreal* ux = su + 2 * (size_t)nz * ny * (nx + 1);
    const wreal* vx = sv + 2 * (size_t)nz * (ny + 1) * nx;
    wreal s = ux[R3(k, j, c, ny, nx + 1)] + ux[R3(k, j, c + 1, ny, nx + 1)];
    if (c == 0) s += ux[R3(k, j, nx, ny, nx + 1)];
    if (c == nx - 1) s += ux[R3(k, j, 0, ny, nx + 1)];
    s += vx[R3(k, j, c, ny + 1, nx)] + vx[R3(k, j + 1, c, ny + 1, nx)];
    if (j == 0) s += vx[R3(k, ny, c, ny + 1, nx)];
    if (j == ny - 1) s += vx[R3(k, 0, c, ny + 1, nx)];
    sg[tid] = (wreal)0.5 * s;
}

#define CC_GATHER_ARGS                                                      \
    const wreal* __restrict__ w, const wreal* __restrict__ mut,             \
    const wreal* __restrict__ su, const wreal* __restrict__ sv,             \
    const wreal* __restrict__ sw, const wreal* __restrict__ sg,             \
    wreal* __restrict__ srw, wreal* __restrict__ aru, wreal* __restrict__ arv, \
    wreal* __restrict__ au, wreal* __restrict__ av, wreal* __restrict__ aw, \
    wreal* __restrict__ amut, CC_PASSIVE, CC_GEOM

// u face (k, j, i): a_u and a_ru.
extern "C" __global__
void cc_ad_u(CC_GATHER_ARGS)
{
    size_t ul = (size_t)ny * (nx + 1);
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)nz * ul) return;
    int k = (int)(tid / ul);
    int j = (int)((tid - (size_t)k * ul) / (nx + 1));
    int i = (int)(tid - (size_t)k * ul - (size_t)j * (nx + 1));
    size_t st = (size_t)ny * nx, nw = (size_t)nz * st;
    wreal gu = su[3 * (size_t)nz * ul + tid];
    wreal gr = (wreal)0;
    // vxgm at mass points i (left face) and i - 1 (right face).
    for (int s = 0; s < 2; ++s) {
        int c = i - s;
        if (c < 0 || c >= nx) continue;
        wreal dmv = msfv[(size_t)(j + 1) * nx + c] - msfv[(size_t)j * nx + c];
        gu += (wreal)0.5 * dmv * rdy * sg[(size_t)k * st + (size_t)j * nx + c];
        // w levels k (fzm) and k + 1 (fzp) of cell (j, c).
        size_t cc = (size_t)j * nx + c;
        if (k >= 1) {
            gu += (wreal)0.5 * fzm[k] * sw[nw + (size_t)k * st + cc];
            gr += (wreal)0.5 * fzm[k] * sw[(size_t)k * st + cc];
        }
        if (k + 1 <= nz - 1) {
            gu += (wreal)0.5 * fzp[k + 1] * sw[nw + (size_t)(k + 1) * st + cc];
            gr += (wreal)0.5 * fzp[k + 1] * sw[(size_t)(k + 1) * st + cc];
        }
        // v faces of column c whose jc or jmc is j.
        wreal r = sv[R3(k, j, c, ny + 1, nx)] + sv[R3(k, j + 1, c, ny + 1, nx)];
        if (j == 0) r += sv[R3(k, ny, c, ny + 1, nx)];
        if (j == ny - 1) r += sv[R3(k, 0, c, ny + 1, nx)];
        gr += (wreal)0.25 * r;
    }
    au[tid] += gu;
    aru[tid] += gr;
}

// v face (k, j, i): a_v and a_rv.
extern "C" __global__
void cc_ad_v(CC_GATHER_ARGS)
{
    size_t vl = (size_t)(ny + 1) * nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)nz * vl) return;
    int k = (int)(tid / vl);
    int j = (int)((tid - (size_t)k * vl) / nx);
    int i = (int)(tid - (size_t)k * vl - (size_t)j * nx);
    size_t st = (size_t)ny * nx, nw = (size_t)nz * st;
    wreal gv = sv[3 * (size_t)nz * vl + tid];
    wreal gr = (wreal)0;
    for (int s = 0; s < 2; ++s) {
        int r = j - s;
        if (r < 0 || r >= ny) continue;
        size_t cr = (size_t)r * nx + i;
        wreal dmu = msfu[(size_t)r * (nx + 1) + i + 1] - msfu[(size_t)r * (nx + 1) + i];
        gv -= (wreal)0.5 * dmu * rdx * sg[(size_t)k * st + cr];
        if (k >= 1) {
            gv += (wreal)0.5 * fzm[k] * sw[3 * nw + (size_t)k * st + cr];
            gr += (wreal)0.5 * fzm[k] * sw[2 * nw + (size_t)k * st + cr];
        }
        if (k + 1 <= nz - 1) {
            gv += (wreal)0.5 * fzp[k + 1] * sw[3 * nw + (size_t)(k + 1) * st + cr];
            gr += (wreal)0.5 * fzp[k + 1] * sw[2 * nw + (size_t)(k + 1) * st + cr];
        }
        // u faces of row r whose ic or imc is i.
        wreal q = su[R3(k, r, i, ny, nx + 1)] + su[R3(k, r, i + 1, ny, nx + 1)];
        if (i == 0) q += su[R3(k, r, nx, ny, nx + 1)];
        if (i == nx - 1) q += su[R3(k, r, 0, ny, nx + 1)];
        gr += (wreal)0.25 * q;
    }
    av[tid] += gv;
    arv[tid] += gr;
}

// w level (kf, j, i): a_rw from the faces' rw4, then a_w; a_rw kept in
// srw (nz+1, ny, nx) for the column mass.
extern "C" __global__
void cc_ad_w(CC_GATHER_ARGS)
{
    size_t st = (size_t)ny * nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)(nz + 1) * st) return;
    int kf = (int)(tid / st);
    size_t c = tid - (size_t)kf * st;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    size_t ul = (size_t)ny * (nx + 1), vl = (size_t)(ny + 1) * nx;
    const wreal* ur = su + (size_t)nz * ul;
    const wreal* vr = sv + (size_t)nz * vl;
    wreal s = (wreal)0;
    for (int d = 0; d < 2; ++d) {
        int k = kf - d;
        if (k < 0 || k >= nz) continue;
        s += ur[R3(k, j, i, ny, nx + 1)] + ur[R3(k, j, i + 1, ny, nx + 1)];
        if (i == 0) s += ur[R3(k, j, nx, ny, nx + 1)];
        if (i == nx - 1) s += ur[R3(k, j, 0, ny, nx + 1)];
        s += vr[R3(k, j, i, ny + 1, nx)] + vr[R3(k, j + 1, i, ny + 1, nx)];
        if (j == 0) s += vr[R3(k, ny, i, ny + 1, nx)];
        if (j == ny - 1) s += vr[R3(k, 0, i, ny + 1, nx)];
    }
    wreal arw = (wreal)0.25 * s;
    srw[tid] = arw;
    aw[tid] += wdiv((c1f[kf] * mut[c] + c2f[kf]) * arw, msft[c]);
}

extern "C" __global__
void cc_ad_mut(CC_GATHER_ARGS)
{
    size_t st = (size_t)ny * nx;
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= st) return;
    wreal s = (wreal)0;
    for (int kf = 0; kf <= nz; ++kf)
        s += c1f[kf] * w[(size_t)kf * st + c] * srw[(size_t)kf * st + c];
    amut[c] += wdiv(s, msft[c]);
}

// ================================================================ w_damp
//
// woof's strict w_damp (woof/core/kernels/openbc.cu, w_damping = 1): on
// interior w levels where vert_cfl = |ww/(c1f mut + c2f) rdnw dt| exceeds
// w_damp_on, rw_t -= sign(1, w) w_alpha (vert_cfl - w_crit_cfl)
// (c1f mut + c2f).  Active: ww and mup (w enters through its sign only).
// The switch is taken on the base state, as WRFPLUS does.

#define WD_PASSIVE                                                          \
    const wreal* __restrict__ w, const wreal* __restrict__ mub,             \
    const wreal* __restrict__ c1f, const wreal* __restrict__ c2f,           \
    const wreal* __restrict__ rdnw

#define WD_GEOM wreal dt, wreal w_damp_on, wreal w_crit_cfl, wreal w_alpha, int nz, int ny, int nx

static __device__ __forceinline__ wreal wd_sign(wreal x)
{
    return signbit(x) ? (wreal)-1 : (wreal)1;
}

extern "C" __global__
void wd_tl(wreal* __restrict__ rw_t, wreal* __restrict__ rw_d,
           const wreal* __restrict__ ww, const wreal* __restrict__ wwd,
           const wreal* __restrict__ mup, const wreal* __restrict__ mupd,
           WD_PASSIVE, int nl, WD_GEOM)
{
    size_t st = (size_t)ny * nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (nz < 2 || tid >= (size_t)(nz - 1) * st) return;
    int k = (int)(tid / st) + 1;
    size_t c = tid - (size_t)(k - 1) * st;
    size_t ix = (size_t)k * st + c;
    Dl m = c1f[k] * (dl(mub[c], (wreal)0) + ld(mup, mupd, c)) + c2f[k];
    wreal q = wdiv(ww[ix], m.v);
    wreal qd = wdiv(wwd[ix] - q * m.d, m.v);
    wreal x = (q * rdnw[k]) * dt;
    wreal vc = fabs(x);
    if (vc > w_damp_on) {
        wreal vcd = wd_sign(x) * ((qd * rdnw[k]) * dt);
        wreal damp = wd_sign(w[ix]) * w_alpha;
        Dl t = (damp * dl(vc - w_crit_cfl, vcd)) * m;
        if (nl) rw_t[ix] = rw_t[ix] - t.v;
        if (WRFPLUS_TL) rw_d[ix] = rw_d[ix] - t.d;
    }
}

// One thread per column: the column's ww words and its mup word.
extern "C" __global__
void wd_ad(const wreal* __restrict__ rw_a, const wreal* __restrict__ ww,
           const wreal* __restrict__ mup, wreal* __restrict__ aww,
           wreal* __restrict__ amup, WD_PASSIVE, WD_GEOM)
{
    size_t st = (size_t)ny * nx;
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= st) return;
    wreal am = (wreal)0;
    for (int k = 1; k <= nz - 1; ++k) {
        size_t ix = (size_t)k * st + c;
        wreal m = c1f[k] * (mub[c] + mup[c]) + c2f[k];
        wreal q = wdiv(ww[ix], m);
        wreal x = (q * rdnw[k]) * dt;
        wreal vc = fabs(x);
        if (!(vc > w_damp_on)) continue;
        wreal sa = wd_sign(w[ix]) * w_alpha;
        wreal a = -rw_a[ix];
        wreal aq = a * sa * m * wd_sign(x) * rdnw[k] * dt;
        aww[ix] += wdiv(aq, m);
        am += c1f[k] * (a * sa * (vc - w_crit_cfl) - wdiv(aq * q, m));
    }
    amup[c] += am;
}
