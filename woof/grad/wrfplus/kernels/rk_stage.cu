// Tangent-linear and adjoint of woof's strict RK-stage glue (P3a item 4):
//
//   calc_p_rho_phi   woof calc_p_alpha / calc_p_alpha_carried_muts
//                    (woof/core/kernels/diagnostics.cu, strict
//                    wrf_p_rho_phi_column), dry, hypsometric_opt 1 or 2;
//   rk_step_prep     woof stage_fluxes: couple_momentum on calc_mu_uv's
//                    face masses (ru, rv) and calc_ww_cp on its own face
//                    masses (ww), woof/core/dycore.py + ieva.py;
//   rk_addtend_dry   woof add_fixed_dry_tendencies (tend += held tendf,
//                    already divided by the map factor) and add_held_heating
//                    (woof/core/kernels/held_heating.cu);
//   set_w_surface    woof set_surface_w (woof/core/kernels/surface_w.cu).
//
// The oracle is WRFPLUS 4.7.1 G_/A_CALC_P_RHO_PHI, G_/A_CALC_MU_UV,
// G_/A_COUPLE_MOMENTUM, g_/a_calc_ww_cp, G_/A_RK_ADDTEND_DRY and
// g_/a_set_w_surface; see docs/dev/P3A-WRF-DRYCORE-ADJOINT.md.
//
// wreal is binary64 unless the unit is built with -DWRFPLUS_REAL=float.
// The base-state statements keep the strict primal's operation order, so
// the float instantiation's base state gives the primal's words.  The
// float build takes glibc_flt32.cuh's gfk_log/gfk_pow (prepended by the
// wrapper), as the strict diagnosis does; binary64 takes log/pow.  Plain
// '+', '-', '*', '/' are the primal's __f*_rn only with --fmad=false and
// IEEE division (never -use_fast_math), which the wrapper passes.
//
// Layout (k, j, i): u-type (nz, ny, nx+1), v-type (nz, ny+1, nx), half
// levels (nz, ny, nx), w levels (nz+1, ny, nx), columns (ny, nx).
// Adjoints: written outputs' adjoints are consumed (zeroed), inputs'
// adjoints are added; every output word has one writer (no atomics).

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

#ifndef RK_KCHUNK
#define RK_KCHUNK 8
#endif

#define R3(k, j, i, nyy, nxx) \
    ((size_t)(k) * (size_t)(nyy) * (size_t)(nxx) + (size_t)(j) * (size_t)(nxx) + (size_t)(i))

static __device__ __forceinline__ double wlog(double x) { return log(x); }
static __device__ __forceinline__ double wpow(double x, double y) { return pow(x, y); }
#ifdef WRFPLUS_FLOAT_LIBM
static __device__ __forceinline__ float wlog(float x) { return gfk_log(x); }
static __device__ __forceinline__ float wpow(float x, float y) { return gfk_pow(x, y); }
#endif

#define RK_RD ((wreal)287.0)
#define RK_P0 ((wreal)100000.0)
#define RK_GAMMA ((wreal)1.4)

// Cells a (right/north) and b (left/south) of face f on an n-cell axis
// (calc_mu_uv: clamped when bnd, wrapped when periodic).
static __device__ __forceinline__ int rk_cell_a(int f, int n, int bnd)
{
    return bnd ? min(f, n - 1) : f % n;
}
static __device__ __forceinline__ int rk_cell_b(int f, int n, int bnd)
{
    return bnd ? max(f - 1, 0) : (f - 1 + n) % n;
}
// Face q (0-3) whose stencil names cell p, one entry per (face, role); -1:
// none.  See small_step.cu ss_face.
static __device__ __forceinline__ int rk_face(int q, int p, int n, int bnd)
{
    if (q == 0) return p;
    if (q == 1) return p + 1;
    if (q == 2) return (bnd ? p == 0 : p == n - 1) ? 0 : -1;
    return (bnd ? p == n - 1 : p == 0) ? n : -1;
}

// ====================================================== calc_p_rho_phi

#define PRP_GEOM int nz, int ny, int nx, int hypso, int base3d, int carried, \
    wreal p_top

#define PRP_STATE(s)                                                        \
    wreal* __restrict__ p##s, wreal* __restrict__ al##s,                   \
    wreal* __restrict__ alt##s, wreal* __restrict__ p_pp##s

#define PRP_ACTIVE(s)                                                       \
    const wreal* __restrict__ thp##s, const wreal* __restrict__ php##s,    \
    const wreal* __restrict__ mup##s, const wreal* __restrict__ muts##s

#define PRP_ADJ                                                             \
    wreal* __restrict__ thp_a, wreal* __restrict__ php_a,                  \
    wreal* __restrict__ mup_a, wreal* __restrict__ muts_a

#define PRP_PASSIVE                                                         \
    const wreal* __restrict__ mub, const wreal* __restrict__ phb,          \
    const wreal* __restrict__ alb, const wreal* __restrict__ pb,           \
    const wreal* __restrict__ rdnw, const wreal* __restrict__ c1h,         \
    const wreal* __restrict__ c2h, const wreal* __restrict__ c3h,          \
    const wreal* __restrict__ c4h, const wreal* __restrict__ c3f,          \
    const wreal* __restrict__ c4f

// One column of calc_p_rho_phi, base state in the strict order; with
// d != 0 also the tangent.  Writes p, al, alt, p_pp (and their tangents).
extern "C" __global__
void calc_p_rho_phi_tl(PRP_STATE(), PRP_STATE(_d), PRP_ACTIVE(), PRP_ACTIVE(_d),
                       PRP_PASSIVE, int nl, PRP_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    size_t st = (size_t)ny * nx;
    size_t kstr = base3d ? st : 1;
    size_t coff = base3d ? (size_t)c : 0;
    wreal M = carried ? muts[c] : mub[c] + mup[c];
    wreal M_d = carried ? muts_d[c] : mup_d[c];
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        size_t b = (size_t)k * kstr + coff;
        wreal dphp = php[h + st] - php[h];
        wreal dphp_d = php_d[h + st] - php_d[h];
        wreal ap, ap_d;
        if (hypso == 2) {
            wreal pfu = (c3f[k + 1] * M + c4f[k + 1]) + p_top;
            wreal pfd = (c3f[k] * M + c4f[k]) + p_top;
            wreal phm = (c3h[k] * M + c4h[k]) + p_top;
            wreal dph = (dphp + phb[b + kstr]) - phb[b];
            wreal L = wlog(pfd / pfu);
            wreal q = dph / phm;
            ap = q / L - alb[b];
            wreal L_d = c3f[k] * M_d / pfd - c3f[k + 1] * M_d / pfu;
            wreal q_d = (dphp_d - q * (c3h[k] * M_d)) / phm;
            ap_d = (q_d - (q / L) * L_d) / L;
        } else {
            wreal den = c1h[k] * M + c2h[k];
            wreal inv = (wreal)-1.0 / den;
            wreal bm = alb[b] * (c1h[k] * mup[c]);
            wreal s = bm + rdnw[k] * dphp;
            ap = inv * s;
            wreal inv_d = c1h[k] * M_d / (den * den);
            wreal s_d = alb[b] * c1h[k] * mup_d[c] + rdnw[k] * dphp_d;
            ap_d = inv_d * s + inv * s_d;
        }
        wreal a = ap + alb[b];
        wreal num = RK_RD * ((wreal)300.0 + thp[h]);
        wreal r = num / (RK_P0 * a);
        wreal pw = wpow(r, RK_GAMMA);
        wreal pp = pw * RK_P0 - pb[b];
        wreal r_d = (RK_RD * thp_d[h] - r * RK_P0 * ap_d) / (RK_P0 * a);
        wreal pp_d = RK_P0 * RK_GAMMA * pw / r * r_d;
        if (WRFPLUS_TL) alt_d[h] = ap_d;
        if (WRFPLUS_TL) al_d[h] = ap_d;
        if (WRFPLUS_TL) p_pp_d[h] = pp_d;
        if (WRFPLUS_TL) p_d[h] = pp_d;
        if (nl) {
            alt[h] = a;
            al[h] = ap;
            p_pp[h] = pp;
            p[h] = pp + pb[b];
        }
    }
}

// Adjoint, one thread per column: consumes p, al, alt and p_pp's
// adjoints; adds into thp, php (own column), mup and muts (or mup when
// not carried).
extern "C" __global__
void calc_p_rho_phi_ad(PRP_STATE(_a), PRP_ACTIVE(), PRP_ADJ, PRP_PASSIVE,
                       PRP_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    size_t st = (size_t)ny * nx;
    size_t kstr = base3d ? st : 1;
    size_t coff = base3d ? (size_t)c : 0;
    wreal M = carried ? muts[c] : mub[c] + mup[c];
    wreal a_M = 0, a_mu = 0;
    wreal a_up = 0;          // php adjoint at level k+1 from layer k
    for (int k = nz - 1; k >= 0; --k) {
        size_t h = (size_t)k * st + c;
        size_t b = (size_t)k * kstr + coff;
        wreal dphp = php[h + st] - php[h];
        wreal ap, den = 0, s = 0, L = 0, q = 0, pfu = 0, pfd = 0, phm = 0;
        if (hypso == 2) {
            pfu = (c3f[k + 1] * M + c4f[k + 1]) + p_top;
            pfd = (c3f[k] * M + c4f[k]) + p_top;
            phm = (c3h[k] * M + c4h[k]) + p_top;
            wreal dph = (dphp + phb[b + kstr]) - phb[b];
            L = wlog(pfd / pfu);
            q = dph / phm;
            ap = q / L - alb[b];
        } else {
            den = c1h[k] * M + c2h[k];
            s = alb[b] * (c1h[k] * mup[c]) + rdnw[k] * dphp;
            ap = -s / den;
        }
        wreal a = ap + alb[b];
        wreal ra = (wreal)1.0 / (RK_P0 * a);
        wreal r = RK_RD * ((wreal)300.0 + thp[h]) * ra;
        wreal pw = wpow(r, RK_GAMMA);

        wreal a_pp = p_a[h] + p_pp_a[h];
        p_a[h] = 0; p_pp_a[h] = 0;
        wreal a_r = a_pp * RK_P0 * RK_GAMMA * pw / r;
        thp_a[h] += a_r * RK_RD * ra;
        wreal a_ap = al_a[h] + alt_a[h] - a_r * r * RK_P0 * ra;
        al_a[h] = 0; alt_a[h] = 0;
        wreal a_dphp;
        if (hypso == 2) {
            wreal a_q = a_ap / L;
            wreal a_L = -a_ap * q / (L * L);
            a_dphp = a_q / phm;
            wreal a_phm = -a_q * q / phm;
            a_M += c3f[k] * a_L / pfd - c3f[k + 1] * a_L / pfu + c3h[k] * a_phm;
        } else {
            wreal rden = (wreal)1.0 / den;
            wreal a_s = -a_ap * rden;
            a_M += a_ap * s * rden * rden * c1h[k];
            a_mu += a_s * alb[b] * c1h[k];
            a_dphp = a_s * rdnw[k];
        }
        php_a[h + st] += a_up + a_dphp;
        a_up = -a_dphp;
    }
    php_a[c] += a_up;
    if (carried) {
        muts_a[c] += a_M;
        mup_a[c] += a_mu;
    } else {
        mup_a[c] += a_mu + a_M;
    }
}

// ======================================================== rk_step_prep

#define RSP_GEOM int nz, int ny, int nx, int has_msf, int boundary_x,        \
    int boundary_y, wreal rdx, wreal rdy

#define RSP_STATE(s)                                                        \
    wreal* __restrict__ ru##s, wreal* __restrict__ rv##s,                  \
    wreal* __restrict__ ww##s

#define RSP_ACTIVE(s)                                                       \
    const wreal* __restrict__ u##s, const wreal* __restrict__ v##s,        \
    const wreal* __restrict__ mup##s

#define RSP_PASSIVE                                                         \
    const wreal* __restrict__ mub, const wreal* __restrict__ c1h,          \
    const wreal* __restrict__ c2h, const wreal* __restrict__ dnw,          \
    const wreal* __restrict__ msfu, const wreal* __restrict__ msfv,        \
    const wreal* __restrict__ msft

// calc_mu_uv's face mass (ieva.stage_face_masses: the two perturbations,
// then the two base words) and calc_ww_cp's (each cell's base and
// perturbation first).  Both have the tangent 0.5*(mup_d[a] + mup_d[b]).
static __device__ __forceinline__ wreal rk_mu_uv(const wreal* mp, const wreal* mb,
                                                 size_t a, size_t b)
{
    return (wreal)0.5 * (((mp[a] + mp[b]) + mb[a]) + mb[b]);
}
static __device__ __forceinline__ wreal rk_mu_ww(const wreal* mp, const wreal* mb,
                                                 size_t a, size_t b)
{
    return (wreal)0.5 * (((mp[a] + mb[a]) + mp[b]) + mb[b]);
}

// couple_momentum: ru = (c1h*muu + c2h)*u [/msfu], rv = (c1h*muv + c2h)*v
// [*msfv, the strict path's msfv_inv]; one thread per (k, face column).
extern "C" __global__
void rk_step_prep_flux_tl(RSP_STATE(), RSP_STATE(_d), RSP_ACTIVE(), RSP_ACTIVE(_d),
                          RSP_PASSIVE, int nl, RSP_GEOM)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;
    if (j < ny) {
        size_t a = (size_t)j * nx + rk_cell_a(i, nx, boundary_x);
        size_t b = (size_t)j * nx + rk_cell_b(i, nx, boundary_x);
        size_t ix = R3(k, j, i, ny, nxf);
        wreal mu = rk_mu_uv(mup, mub, a, b);
        wreal mu_d = (wreal)0.5 * (mup_d[a] + mup_d[b]);
        wreal C = c1h[k] * mu + c2h[k];
        wreal f = C * u[ix];
        wreal f_d = c1h[k] * mu_d * u[ix] + C * u_d[ix];
        if (has_msf) {
            wreal m = msfu[(size_t)j * nxf + i];
            f = f / m;
            f_d = f_d / m;
        }
        if (WRFPLUS_TL) ru_d[ix] = f_d;
        if (nl) ru[ix] = f;
    }
    if (i < nx) {
        size_t a = (size_t)rk_cell_a(j, ny, boundary_y) * nx + i;
        size_t b = (size_t)rk_cell_b(j, ny, boundary_y) * nx + i;
        size_t ix = R3(k, j, i, nyf, nx);
        wreal mu = rk_mu_uv(mup, mub, a, b);
        wreal mu_d = (wreal)0.5 * (mup_d[a] + mup_d[b]);
        wreal C = c1h[k] * mu + c2h[k];
        wreal f = C * v[ix];
        wreal f_d = c1h[k] * mu_d * v[ix] + C * v_d[ix];
        if (has_msf) {
            wreal m = msfv[(size_t)j * nx + i];
            f = f * m;
            f_d = f_d * m;
        }
        if (WRFPLUS_TL) rv_d[ix] = f_d;
        if (nl) rv[ix] = f;
    }
}

// calc_ww_cp's coupled u flux at face column (j, i) and level k.
static __device__ __forceinline__ void rk_fu(const wreal* u, const wreal* u_d,
    const wreal* mup, const wreal* mup_d, const wreal* mub, const wreal* c1h,
    const wreal* c2h, const wreal* msfu, int k, int j, int i, int ny, int nx,
    int has_msf, int bx, wreal* f, wreal* f_d)
{
    int nxf = nx + 1;
    size_t a = (size_t)j * nx + rk_cell_a(i, nx, bx);
    size_t b = (size_t)j * nx + rk_cell_b(i, nx, bx);
    size_t ix = R3(k, j, i, ny, nxf);
    wreal mu = rk_mu_ww(mup, mub, a, b);
    wreal C = c1h[k] * mu + c2h[k];
    *f = C * u[ix];
    *f_d = c1h[k] * ((wreal)0.5 * (mup_d[a] + mup_d[b])) * u[ix] + C * u_d[ix];
    if (has_msf) {
        wreal m = msfu[(size_t)j * nxf + i];
        *f = *f / m;
        *f_d = *f_d / m;
    }
}
static __device__ __forceinline__ void rk_fv(const wreal* v, const wreal* v_d,
    const wreal* mup, const wreal* mup_d, const wreal* mub, const wreal* c1h,
    const wreal* c2h, const wreal* msfv, int k, int j, int i, int ny, int nx,
    int has_msf, int by, wreal* f, wreal* f_d)
{
    int nyf = ny + 1;
    size_t a = (size_t)rk_cell_a(j, ny, by) * nx + i;
    size_t b = (size_t)rk_cell_b(j, ny, by) * nx + i;
    size_t ix = R3(k, j, i, nyf, nx);
    wreal mu = rk_mu_ww(mup, mub, a, b);
    wreal C = c1h[k] * mu + c2h[k];
    *f = C * v[ix];
    *f_d = c1h[k] * ((wreal)0.5 * (mup_d[a] + mup_d[b])) * v[ix] + C * v_d[ix];
    if (has_msf) {
        wreal m = msfv[(size_t)j * nx + i];
        *f = *f * m;
        *f_d = *f_d * m;
    }
}

// calc_ww_cp, one thread per column (woof _omega_column_kernel's order).
extern "C" __global__
void rk_step_prep_ww_tl(RSP_STATE(), RSP_STATE(_d), RSP_ACTIVE(), RSP_ACTIVE(_d),
                        RSP_PASSIVE, int nl, RSP_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    int j = c / nx, i = c - j * nx;
    size_t st = (size_t)ny * nx;
    wreal dmdt = 0, dmdt_d = 0;
    for (int k = 0; k < nz; ++k) {
        wreal f0, f0d, f1, f1d, g0, g0d, g1, g1d;
        rk_fu(u, u_d, mup, mup_d, mub, c1h, c2h, msfu, k, j, i, ny, nx, has_msf,
              boundary_x, &f0, &f0d);
        rk_fu(u, u_d, mup, mup_d, mub, c1h, c2h, msfu, k, j, i + 1, ny, nx, has_msf,
              boundary_x, &f1, &f1d);
        rk_fv(v, v_d, mup, mup_d, mub, c1h, c2h, msfv, k, j, i, ny, nx, has_msf,
              boundary_y, &g0, &g0d);
        rk_fv(v, v_d, mup, mup_d, mub, c1h, c2h, msfv, k, j + 1, i, ny, nx, has_msf,
              boundary_y, &g1, &g1d);
        wreal W = has_msf ? msft[c] * dnw[k] : dnw[k];
        wreal divv = W * (rdx * (f1 - f0) + rdy * (g1 - g0));
        wreal divv_d = W * (rdx * (f1d - f0d) + rdy * (g1d - g0d));
        dmdt = dmdt + divv;
        dmdt_d = dmdt_d + divv_d;
        if (WRFPLUS_TL) ww_d[(size_t)(k + 1) * st + c] = divv_d;
        if (nl) ww[(size_t)(k + 1) * st + c] = divv;
    }
    wreal w = 0, w_d = 0;
    if (WRFPLUS_TL) ww_d[c] = 0;
    if (nl) ww[c] = 0;
    for (int k = 1; k < nz; ++k) {
        size_t at = (size_t)k * st + c;
        wreal divv_d = ww_d[at];
        w_d = (w_d - (dnw[k - 1] * c1h[k - 1]) * dmdt_d) - divv_d;
        if (WRFPLUS_TL) ww_d[at] = w_d;
        if (nl) {
            wreal divv = ww[at];
            w = (w - (dnw[k - 1] * c1h[k - 1]) * dmdt) - divv;
            ww[at] = w;
        }
    }
    if (WRFPLUS_TL) ww_d[(size_t)nz * st + c] = 0;
    if (nl) ww[(size_t)nz * st + c] = 0;
}

// Adjoint of calc_ww_cp's column recurrence: consumes ww's adjoint and
// leaves bk(k) = the adjoint of layer k's bracket (rdx*du + rdy*dv).
extern "C" __global__
void rk_step_prep_ww_ad(RSP_STATE(_a), wreal* __restrict__ bk, RSP_PASSIVE,
                        RSP_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    size_t st = (size_t)ny * nx;
    // a_divv(k-1) = -S_k, S_k = sum_{m >= k, m < nz} a_ww(m); dmdt's
    // adjoint -sum_k S_k dnw(k-1) c1h(k-1) reaches every layer.
    wreal S = 0, a_dmdt = 0;
    for (int k = nz - 1; k >= 1; --k) {
        size_t at = (size_t)k * st + c;
        S += ww_a[at];
        ww_a[at] = 0;
        a_dmdt -= S * (dnw[k - 1] * c1h[k - 1]);
        bk[(size_t)(k - 1) * st + c] = -S;
    }
    ww_a[c] = 0;
    ww_a[(size_t)nz * st + c] = 0;
    bk[(size_t)(nz - 1) * st + c] = 0;
    for (int k = 0; k < nz; ++k) {
        size_t at = (size_t)k * st + c;
        wreal W = has_msf ? msft[c] * dnw[k] : dnw[k];
        bk[at] = W * (bk[at] + a_dmdt);
    }
}

// Face pass of the adjoint, one thread per (level chunk, face column):
// consumes ru/rv's adjoints, adds into u and v, and leaves each face's
// chunk sum of the face-mass adjoint (xm, ym) for the cell gather.
extern "C" __global__
void rk_step_prep_flux_ad(RSP_STATE(_a), const wreal* __restrict__ bk,
                          RSP_ACTIVE(), wreal* __restrict__ u_a,
                          wreal* __restrict__ v_a, wreal* __restrict__ xm,
                          wreal* __restrict__ ym, RSP_PASSIVE, RSP_GEOM)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    int nck = (nz + RK_KCHUNK - 1) / RK_KCHUNK;
    if (tid >= nck * nyf * nxf) return;
    int ck = tid / (nyf * nxf);
    int r = tid - ck * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;
    int k0 = ck * RK_KCHUNK, k1 = min(nz, k0 + RK_KCHUNK);
    size_t st = (size_t)ny * nx;
    if (j < ny) {
        size_t a = (size_t)j * nx + rk_cell_a(i, nx, boundary_x);
        size_t b = (size_t)j * nx + rk_cell_b(i, nx, boundary_x);
        wreal Cr = rk_mu_uv(mup, mub, a, b), Cw = rk_mu_ww(mup, mub, a, b);
        wreal rm = has_msf ? (wreal)1.0 / msfu[(size_t)j * nxf + i] : (wreal)1.0;
        wreal am = 0;
        for (int k = k0; k < k1; ++k) {
            size_t ix = R3(k, j, i, ny, nxf);
            wreal af = 0;     // du(cell i-1) = f(i) - f(i-1); du(cell i) = f(i+1) - f(i)
            if (i > 0) af += rdx * bk[(size_t)k * st + (size_t)j * nx + i - 1];
            if (i < nx) af -= rdx * bk[(size_t)k * st + (size_t)j * nx + i];
            af *= rm;
            wreal ar = ru_a[ix] * rm;
            ru_a[ix] = 0;
            u_a[ix] += ar * (c1h[k] * Cr + c2h[k]) + af * (c1h[k] * Cw + c2h[k]);
            am += (ar + af) * u[ix] * c1h[k];
        }
        xm[(size_t)ck * ny * nxf + (size_t)j * nxf + i] = am;
    }
    if (i < nx) {
        size_t a = (size_t)rk_cell_a(j, ny, boundary_y) * nx + i;
        size_t b = (size_t)rk_cell_b(j, ny, boundary_y) * nx + i;
        wreal Cr = rk_mu_uv(mup, mub, a, b), Cw = rk_mu_ww(mup, mub, a, b);
        wreal m = has_msf ? msfv[(size_t)j * nx + i] : (wreal)1.0;
        wreal am = 0;
        for (int k = k0; k < k1; ++k) {
            size_t ix = R3(k, j, i, nyf, nx);
            wreal af = 0;
            if (j > 0) af += rdy * bk[(size_t)k * st + (size_t)(j - 1) * nx + i];
            if (j < ny) af -= rdy * bk[(size_t)k * st + (size_t)j * nx + i];
            af *= m;
            wreal ar = rv_a[ix] * m;
            rv_a[ix] = 0;
            v_a[ix] += ar * (c1h[k] * Cr + c2h[k]) + af * (c1h[k] * Cw + c2h[k]);
            am += (ar + af) * v[ix] * c1h[k];
        }
        ym[(size_t)ck * nyf * nx + (size_t)j * nx + i] = am;
    }
}

// Cell gather: mup_a += 0.5 * (chunk sums of every face naming the cell).
extern "C" __global__
void rk_step_prep_mass_gather(wreal* __restrict__ mup_a, const wreal* __restrict__ xm,
                              const wreal* __restrict__ ym, RSP_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    int j = c / nx, i = c - j * nx;
    int nyf = ny + 1, nxf = nx + 1;
    int nck = (nz + RK_KCHUNK - 1) / RK_KCHUNK;
    wreal s = 0;
    for (int ck = 0; ck < nck; ++ck) {
        for (int q = 0; q < 4; ++q) {
            int f = rk_face(q, i, nx, boundary_x);
            if (f >= 0) s += xm[(size_t)ck * ny * nxf + (size_t)j * nxf + f];
            int g = rk_face(q, j, ny, boundary_y);
            if (g >= 0) s += ym[(size_t)ck * nyf * nx + (size_t)g * nx + i];
        }
    }
    mup_a[c] += (wreal)0.5 * s;
}

// ====================================================== rk_addtend_dry

#define AT_GEOM int nz, int ny, int nx, int has_msf, int heating

#define AT_STATE(s)                                                         \
    wreal* __restrict__ ru_t##s, wreal* __restrict__ rv_t##s,              \
    wreal* __restrict__ rw_t##s, wreal* __restrict__ rph_t##s,             \
    wreal* __restrict__ rth_t##s, wreal* __restrict__ rmu_t##s

#define AT_ACTIVE(s)                                                        \
    const wreal* __restrict__ ru_tf##s, const wreal* __restrict__ rv_tf##s, \
    const wreal* __restrict__ rw_tf##s, const wreal* __restrict__ rph_tf##s, \
    const wreal* __restrict__ rth_tf##s, const wreal* __restrict__ rmu_tf##s, \
    const wreal* __restrict__ h_diabatic##s, const wreal* __restrict__ mup##s

#define AT_ADJ                                                              \
    wreal* __restrict__ ru_tf_a, wreal* __restrict__ rv_tf_a,              \
    wreal* __restrict__ rw_tf_a, wreal* __restrict__ rph_tf_a,             \
    wreal* __restrict__ rth_tf_a, wreal* __restrict__ rmu_tf_a,            \
    wreal* __restrict__ h_diabatic_a, wreal* __restrict__ mup_a

#define AT_PASSIVE                                                          \
    const wreal* __restrict__ mub, const wreal* __restrict__ c1h,          \
    const wreal* __restrict__ c2h, const wreal* __restrict__ msft

// tend += tendf for the six dry tendencies; then (heating) rth_t +=
// (c1h*(mub+mup) + c2h)*h_diabatic [/msft].  One thread per index of the
// largest (w-level/face) extent.
extern "C" __global__
void rk_addtend_dry_tl(AT_STATE(), AT_STATE(_d), AT_ACTIVE(), AT_ACTIVE(_d),
                       AT_PASSIVE, int nl, AT_GEOM)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t nu = (size_t)nz * ny * (nx + 1), nv = (size_t)nz * (ny + 1) * nx;
    size_t nw = (size_t)(nz + 1) * ny * nx, nm = (size_t)nz * ny * nx;
    size_t st = (size_t)ny * nx;
    if (t < nu) { if (WRFPLUS_TL) ru_t_d[t] += ru_tf_d[t]; if (nl) ru_t[t] = ru_t[t] + ru_tf[t]; }
    if (t < nv) { if (WRFPLUS_TL) rv_t_d[t] += rv_tf_d[t]; if (nl) rv_t[t] = rv_t[t] + rv_tf[t]; }
    if (t < nw) {
        if (WRFPLUS_TL) { rw_t_d[t] += rw_tf_d[t]; rph_t_d[t] += rph_tf_d[t]; }
        if (nl) { rw_t[t] = rw_t[t] + rw_tf[t]; rph_t[t] = rph_t[t] + rph_tf[t]; }
    }
    if (t < st) { if (WRFPLUS_TL) rmu_t_d[t] += rmu_tf_d[t]; if (nl) rmu_t[t] = rmu_t[t] + rmu_tf[t]; }
    if (t < nm) {
        int c = (int)(t % st);
        int k = (int)(t / st);
        wreal th = rth_t[t] + rth_tf[t];
        wreal th_d = rth_t_d[t] + rth_tf_d[t];
        if (heating) {
            wreal mu = mub[c] + mup[c];
            wreal C = c1h[k] * mu + c2h[k];
            wreal hd = C * h_diabatic[t];
            wreal hd_d = c1h[k] * mup_d[c] * h_diabatic[t] + C * h_diabatic_d[t];
            if (has_msf) {
                hd = hd / msft[c];
                hd_d = hd_d / msft[c];
            }
            th = th + hd;
            th_d = th_d + hd_d;
        }
        if (WRFPLUS_TL) rth_t_d[t] = th_d;
        if (nl) rth_t[t] = th;
    }
}

// Adjoint, elementwise part: the tendencies' adjoints pass through (the
// update is tend += ...) and add into tendf and h_diabatic.
extern "C" __global__
void rk_addtend_dry_ad(AT_STATE(_a), AT_ACTIVE(), AT_ADJ, AT_PASSIVE, AT_GEOM)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t nu = (size_t)nz * ny * (nx + 1), nv = (size_t)nz * (ny + 1) * nx;
    size_t nw = (size_t)(nz + 1) * ny * nx, nm = (size_t)nz * ny * nx;
    size_t st = (size_t)ny * nx;
    if (t < nu) ru_tf_a[t] += ru_t_a[t];
    if (t < nv) rv_tf_a[t] += rv_t_a[t];
    if (t < nw) { rw_tf_a[t] += rw_t_a[t]; rph_tf_a[t] += rph_t_a[t]; }
    if (t < st) rmu_tf_a[t] += rmu_t_a[t];
    if (t < nm) {
        rth_tf_a[t] += rth_t_a[t];
        if (heating) {
            int c = (int)(t % st);
            int k = (int)(t / st);
            wreal a = rth_t_a[t];
            if (has_msf) a = a / msft[c];
            h_diabatic_a[t] += a * (c1h[k] * (mub[c] + mup[c]) + c2h[k]);
        }
    }
}

// Adjoint, column part: mup's adjoint from the heating's coupling.
extern "C" __global__
void rk_addtend_dry_ad_mass(AT_STATE(_a), AT_ACTIVE(), AT_ADJ, AT_PASSIVE, AT_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx || !heating) return;
    size_t st = (size_t)ny * nx;
    wreal s = 0;
    for (int k = 0; k < nz; ++k) {
        size_t t = (size_t)k * st + c;
        s += rth_t_a[t] * h_diabatic[t] * c1h[k];
    }
    if (has_msf) s = s / msft[c];
    mup_a[c] += s;
}

// ======================================================= set_w_surface

#define SW_GEOM int nz, int ny, int nx, int mapped, int boundary_x,         \
    int boundary_y, wreal cf1, wreal cf2, wreal cf3, wreal half_rdx,        \
    wreal half_rdy

// The strict primal's terrain slopes and coefficients for cell (j, i).
struct SwCell { wreal cy, cx, dyn, dys, dxe, dxw; };

static __device__ __forceinline__ SwCell sw_cell(const wreal* ht, const wreal* msft,
                                                 int j, int i, SW_GEOM)
{
    SwCell s;
    int col = j * nx + i;
    s.dyn = ht[((j + 1) % ny) * nx + i] - ht[col];
    s.dys = ht[col] - ht[((j + ny - 1) % ny) * nx + i];
    s.dxe = ht[j * nx + (i + 1) % nx] - ht[col];
    s.dxw = ht[col] - ht[j * nx + (i + nx - 1) % nx];
    if (boundary_y && j == 0) s.dys = 0;
    if (boundary_y && j == ny - 1) s.dyn = 0;
    if (boundary_x && i == 0) s.dxw = 0;
    if (boundary_x && i == nx - 1) s.dxe = 0;
    wreal msf = mapped ? msft[col] : (wreal)1.0;
    s.cy = (msf * (wreal)0.5) * ((wreal)2.0 * half_rdy);
    s.cx = (msf * (wreal)0.5) * ((wreal)2.0 * half_rdx);
    return s;
}

// w(k=0) = cy*(dyn*vc(j+1) + dys*vc(j)) + cx*(dxe*uc(i+1) + dxw*uc(i)),
// uc = cf1*u(0) + cf2*u(1) + cf3*u(2); one thread per column.
extern "C" __global__
void set_w_surface_tl(wreal* __restrict__ w, wreal* __restrict__ w_d,
                      const wreal* __restrict__ u, const wreal* __restrict__ v,
                      const wreal* __restrict__ u_d, const wreal* __restrict__ v_d,
                      const wreal* __restrict__ ht, const wreal* __restrict__ msft,
                      int nl, SW_GEOM)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    int j = col / nx, i = col - j * nx;
    size_t up = (size_t)ny * (nx + 1), vp = (size_t)(ny + 1) * nx;
    size_t ul = (size_t)j * (nx + 1) + i, vb = (size_t)j * nx + i;
    SwCell s = sw_cell(ht, msft, j, i, nz, ny, nx, mapped, boundary_x,
                       boundary_y, cf1, cf2, cf3, half_rdx, half_rdy);
#define SW_C(a, o) ((cf1 * a[o] + cf2 * a[up + o]) + cf3 * a[2 * up + o])
#define SW_V(a, o) ((cf1 * a[o] + cf2 * a[vp + o]) + cf3 * a[2 * vp + o])
    wreal y_d = s.cy * (s.dyn * SW_V(v_d, vb + nx) + s.dys * SW_V(v_d, vb));
    wreal x_d = s.cx * (s.dxe * SW_C(u_d, ul + 1) + s.dxw * SW_C(u_d, ul));
    if (WRFPLUS_TL) w_d[col] = y_d + x_d;
    if (nl) {
        wreal y = s.cy * (s.dyn * SW_V(v, vb + nx) + s.dys * SW_V(v, vb));
        wreal x = s.cx * (s.dxe * SW_C(u, ul + 1) + s.dxw * SW_C(u, ul));
        w[col] = y + x;
    }
#undef SW_C
#undef SW_V
}

// Adjoint, one thread per face column: consumes w(0)'s adjoint after
// every face has read it (the wrapper zeroes it in a second launch) and
// adds into u and v levels 0-2 from the one or two cells that read the
// face.
extern "C" __global__
void set_w_surface_ad(const wreal* __restrict__ w_a, wreal* __restrict__ u_a,
                      wreal* __restrict__ v_a, const wreal* __restrict__ ht,
                      const wreal* __restrict__ msft, SW_GEOM)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nyf * nxf) return;
    int j = tid / nxf, i = tid - j * nxf;
    size_t up = (size_t)ny * nxf, vp = (size_t)nyf * nx;
    if (j < ny) {
        wreal a = 0;
        if (i < nx) {      // face i is cell i's west face (uc0)
            SwCell s = sw_cell(ht, msft, j, i, nz, ny, nx, mapped, boundary_x,
                               boundary_y, cf1, cf2, cf3, half_rdx, half_rdy);
            a += w_a[j * nx + i] * s.cx * s.dxw;
        }
        if (i > 0) {       // and cell i-1's east face (uc1)
            SwCell s = sw_cell(ht, msft, j, i - 1, nz, ny, nx, mapped, boundary_x,
                               boundary_y, cf1, cf2, cf3, half_rdx, half_rdy);
            a += w_a[j * nx + i - 1] * s.cx * s.dxe;
        }
        size_t o = (size_t)j * nxf + i;
        u_a[o] += cf1 * a;
        u_a[up + o] += cf2 * a;
        u_a[2 * up + o] += cf3 * a;
    }
    if (i < nx) {
        wreal a = 0;
        if (j < ny) {
            SwCell s = sw_cell(ht, msft, j, i, nz, ny, nx, mapped, boundary_x,
                               boundary_y, cf1, cf2, cf3, half_rdx, half_rdy);
            a += w_a[j * nx + i] * s.cy * s.dys;
        }
        if (j > 0) {
            SwCell s = sw_cell(ht, msft, j - 1, i, nz, ny, nx, mapped, boundary_x,
                               boundary_y, cf1, cf2, cf3, half_rdx, half_rdy);
            a += w_a[(j - 1) * nx + i] * s.cy * s.dyn;
        }
        size_t o = (size_t)j * nx + i;
        v_a[o] += cf1 * a;
        v_a[vp + o] += cf2 * a;
        v_a[2 * vp + o] += cf3 * a;
    }
}

extern "C" __global__
void set_w_surface_ad_consume(wreal* __restrict__ w_a, int n)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) w_a[c] = 0;
}
