// Tangent-linear and adjoint of woof's strict advance_w_phi
// (woof/core/kernels/acoustic.cu, the GPUWM_WRF_EXACT branch) and of
// advance_w_phi_msf (the same, with WRF advance_w's msfty factors, when
// has_msf; msftx = msfty = msft, as the primal carries one factor),
// which is WRF 4.7.1 advance_w followed by calc_p_rho's non-hydrostatic
// diagnosis (dyn_em/module_small_step_em.F).  The oracle is WRFPLUS 4.7.1
// g_/a_advance_w and g_/a_calc_p_rho (wrftladj/module_small_step_em_
// {tl,ad}.F); see docs/dev/P3A-WRF-DRYCORE-ADJOINT.md.
//
// wreal is binary64 unless the unit is built with -DWRFPLUS_REAL=float.
// The base-state statements keep the primal's operation order, so the
// float instantiation of advance_w_tl's base state reproduces the strict
// primal's words.  The float build takes glibc_trig_flt32.cuh's glibc_sinf
// (MIT-licensed Arm optimized-routines; prepended by the wrapper) for the
// damper, as the strict primal does; binary64 takes sin.
//
// damp_opt = 3 (dampmag > 0): WRF advance_w's implicit Rayleigh w damper
// (module_small_step_em.F:1445), after the back substitution and before
// the phi'' update.  Per column, htop = (phb + php)(nz)/g, hbot = htop -
// zdamp; at each level k >= 1 with hk = (phb + php)(k)/g >= hbot,
// dw = dampmag*sin^2(pi/2 (hk - hbot)/zdamp) and
// w'' = (w'' - dw (c1f mut + c2f) w_ref)/(1 + dw).  Active in it: w'',
// php (the heights: WRF's ph_1, woof's ph_save), mup (mut) and w_ref
// (WRF w_save, woof's state.w).  Passive: phb, c1f, c2f, dampmag, zdamp.
//
// Layout: woof's (k, j, i).  w levels (nz+1, ny, nx), mass (nz, ny, nx),
// u faces (nz, ny, nx+1), v faces (nz, ny+1, nx), 2-D (ny, nx).  One thread
// per column; columns outside the spec_zone frame are left untouched.
//
// Active inputs: w_pp, ph_pp, rw_t, rph_t, ww_pp, mu_pp, mu_pp_old, th_pp,
// th_pp_old, thp, php, alt, c2a, a, alpha, gam, mup, u_pp, v_pp, cqw (when
// moist_cq), w_ref (when dampmag > 0).  Outputs: w_pp, ph_pp, p_pp, al_pp.  Passive: phb, ht, rdn,
// rdnw, fnm, fnp, c1h, c2h, c1f, c2f, mub2d, msft and the scalars.  With
// has_msf == 0 the factors are an exact 1 and msft is not read, so the
// words are the msf-free kernel's.

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

#ifndef WV_LEV
#define WV_LEV 129
#endif

#define WG ((wreal)9.81)
#define WPI2 ((wreal)1.5707963267948966)

static __device__ __forceinline__ double wsin(double x) { return sin(x); }
static __device__ __forceinline__ double wcos(double x) { return cos(x); }
#ifdef WRFPLUS_FLOAT_TRIG
static __device__ __forceinline__ float wsin(float x) { return glibc_sinf(x); }
static __device__ __forceinline__ float wcos(float x) { return cosf(x); }
#endif
#define WT0 ((wreal)300.0)
#define W3S(k, j, i, ny, nxs) ((size_t)(k) * (ny) * (nxs) + (size_t)(j) * (nxs) + (i))

#define W_TRAJ_PARAMS                                                       \
    const wreal* __restrict__ rw_t, const wreal* __restrict__ rph_t,       \
    const wreal* __restrict__ ww_pp, const wreal* __restrict__ mu_pp,      \
    const wreal* __restrict__ mu_pp_old, const wreal* __restrict__ th_pp,  \
    const wreal* __restrict__ th_pp_old, const wreal* __restrict__ thp,    \
    const wreal* __restrict__ php, const wreal* __restrict__ phb,          \
    const wreal* __restrict__ alt, const wreal* __restrict__ c2a,          \
    const wreal* __restrict__ a, const wreal* __restrict__ alpha,          \
    const wreal* __restrict__ gam, const wreal* __restrict__ mup,          \
    const wreal* __restrict__ u_pp, const wreal* __restrict__ v_pp,        \
    const wreal* __restrict__ ht, const wreal* __restrict__ rdn,           \
    const wreal* __restrict__ rdnw, const wreal* __restrict__ fnm,         \
    const wreal* __restrict__ fnp, const wreal* __restrict__ c1h,          \
    const wreal* __restrict__ c2h, const wreal* __restrict__ c1f,          \
    const wreal* __restrict__ c2f, const wreal* __restrict__ mub2d,        \
    const wreal* __restrict__ cqw, const wreal* __restrict__ msft,         \
    const wreal* __restrict__ w_ref

#define W_GEOM_PARAMS                                                       \
    int nz, int ny, int nx, int spec_zone, int base3d, int top_lid,        \
    int moist_cq, int boundary_x, int boundary_y, wreal cf1, wreal cf2,    \
    wreal cf3, wreal rdx, wreal rdy, wreal dtau, wreal epssm, int has_msf, \
    wreal dampmag, wreal zdamp

// WRF's t_2ave (acoustic.cu wrf_theta_average, strict).
static __device__ __forceinline__
wreal t2ave_(wreal cur, wreal prev, wreal muave, wreal c1, wreal c2,
             wreal muts, wreal theta, wreal epssm)
{
    wreal average = (wreal)0.5 * (((wreal)1.0 + epssm) * cur
                                  + ((wreal)1.0 - epssm) * prev);
    return (average + (c1 * muave) * WT0) / ((c1 * muts + c2) * (WT0 + theta));
}

// The bottom kinematic condition's neighbour columns and face offsets.
struct WBC { size_t cip, cim, cjp, cjm, ue, uw, vn, vs; };

static __device__ __forceinline__
WBC w_bc(int j, int i, int ny, int nx, int boundary_x, int boundary_y)
{
    WBC b;
    int ip = boundary_x ? min(i + 1, nx - 1) : (i + 1) % nx;
    int im = boundary_x ? max(i - 1, 0) : (i - 1 + nx) % nx;
    int jp = boundary_y ? min(j + 1, ny - 1) : (j + 1) % ny;
    int jm = boundary_y ? max(j - 1, 0) : (j - 1 + ny) % ny;
    b.cip = (size_t)j * nx + ip; b.cim = (size_t)j * nx + im;
    b.cjp = (size_t)jp * nx + i; b.cjm = (size_t)jm * nx + i;
    b.ue = W3S(0, j, i + 1, ny, nx + 1); b.uw = W3S(0, j, i, ny, nx + 1);
    b.vn = W3S(0, j + 1, i, ny + 1, nx); b.vs = W3S(0, j, i, ny + 1, nx);
    return b;
}

// w''(0) from the lowest three u/v levels (linear in u, v).
static __device__ __forceinline__
wreal w_bottom(const wreal* u, const wreal* v, const wreal* ht, size_t c,
               const WBC& b, size_t lu, size_t lv, wreal cf1, wreal cf2,
               wreal cf3, wreal rdx, wreal rdy, wreal msf)
{
    wreal ue = cf1 * u[b.ue] + cf2 * u[b.ue + lu] + cf3 * u[b.ue + 2 * lu];
    wreal uw = cf1 * u[b.uw] + cf2 * u[b.uw + lu] + cf3 * u[b.uw + 2 * lu];
    wreal vn = cf1 * v[b.vn] + cf2 * v[b.vn + lv] + cf3 * v[b.vn + 2 * lv];
    wreal vs = cf1 * v[b.vs] + cf2 * v[b.vs + lv] + cf3 * v[b.vs + 2 * lv];
    return msf * (wreal)0.5 * rdy * ((ht[b.cjp] - ht[c]) * vn + (ht[c] - ht[b.cjm]) * vs)
         + msf * (wreal)0.5 * rdx * ((ht[b.cip] - ht[c]) * ue + (ht[c] - ht[b.cim]) * uw);
}

// Joint base-state and tangent-linear update (WRFPLUS g_advance_w then
// g_calc_p_rho at step 0).  With nl == 0 the base state is left unchanged.
extern "C" __global__
void advance_w_tl(wreal* __restrict__ w_pp, wreal* __restrict__ ph_pp,
                  wreal* __restrict__ p_pp, wreal* __restrict__ al_pp,
                  wreal* __restrict__ w_d, wreal* __restrict__ ph_d,
                  wreal* __restrict__ p_d, wreal* __restrict__ al_d,
                  W_TRAJ_PARAMS,
                  const wreal* __restrict__ rw_d, const wreal* __restrict__ rph_d,
                  const wreal* __restrict__ ww_d, const wreal* __restrict__ mupp_d,
                  const wreal* __restrict__ mupold_d, const wreal* __restrict__ th_d,
                  const wreal* __restrict__ thold_d, const wreal* __restrict__ thp_d,
                  const wreal* __restrict__ php_d, const wreal* __restrict__ alt_d,
                  const wreal* __restrict__ c2a_d, const wreal* __restrict__ a_d,
                  const wreal* __restrict__ alpha_d, const wreal* __restrict__ gam_d,
                  const wreal* __restrict__ mup_d, const wreal* __restrict__ u_d,
                  const wreal* __restrict__ v_d, const wreal* __restrict__ cqw_d,
                  const wreal* __restrict__ wref_d,
                  int nl, W_GEOM_PARAMS)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx || nz + 1 > WV_LEV) return;
    int j = c / nx, i = c - j * nx;
    if (i < spec_zone || i >= nx - spec_zone
        || j < spec_zone || j >= ny - spec_zone) return;
    size_t st = (size_t)ny * nx;
    size_t bstr = base3d ? st : 1;
    size_t boff = base3d ? (size_t)c : 0;
    size_t lu = (size_t)ny * (nx + 1), lv = (size_t)(ny + 1) * nx;
    wreal e = epssm;
    // WRF advance_w's msfty and msft_inv = 1/msfty (exact 1 without msf).
    wreal msf = has_msf ? msft[c] : (wreal)1.0;
    wreal msf_i = (wreal)1.0 / msf;

    wreal mut = mub2d[c] + mup[c];
    wreal muts = mut + mu_pp[c];
    wreal muave = (wreal)0.5 * (((wreal)1.0 + e) * mu_pp[c]
                              + ((wreal)1.0 - e) * mu_pp_old[c]);
    wreal mut_d = mup_d[c];
    wreal muts_d = mup_d[c] + mupp_d[c];
    wreal muave_d = (wreal)0.5 * (((wreal)1.0 + e) * mupp_d[c]
                                + ((wreal)1.0 - e) * mupold_d[c]);

    wreal wl[WV_LEV], wld[WV_LEV], phl[WV_LEV], phld[WV_LEV];
    for (int k = 0; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        wl[k] = w_pp[f]; wld[k] = w_d[f];
        phl[k] = ph_pp[f]; phld[k] = ph_d[f];
    }

    // RHS of the phi'' equation.
    wreal rhs[WV_LEV], rhsd[WV_LEV];
    rhs[0] = 0; rhsd[0] = 0;
    for (int k = 1; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        rhs[k] = dtau * (rph_t[f] + (wreal)0.5 * WG * ((wreal)1.0 - e) * wl[k]);
        rhsd[k] = dtau * (rph_d[f] + (wreal)0.5 * WG * ((wreal)1.0 - e) * wld[k]);
    }
    wreal dlo = (php[st + c] - php[c] + phb[bstr + boff] - phb[boff]);
    wreal dlo_d = php_d[st + c] - php_d[c];
    wreal wd_lo = (wreal)0.5 * (ww_pp[st + c] + ww_pp[c]) * rdnw[0] * dlo;
    wreal wd_lo_d = (wreal)0.5 * rdnw[0] * ((ww_d[st + c] + ww_d[c]) * dlo
                                          + (ww_pp[st + c] + ww_pp[c]) * dlo_d);
    for (int k = 1; k < nz; ++k) {
        size_t f = (size_t)k * st + c, fp = f + st;
        wreal dhi = (php[fp] - php[f]
                     + phb[(size_t)(k + 1) * bstr + boff] - phb[(size_t)k * bstr + boff]);
        wreal dhi_d = php_d[fp] - php_d[f];
        wreal wd_hi = (wreal)0.5 * (ww_pp[fp] + ww_pp[f]) * rdnw[k] * dhi;
        wreal wd_hi_d = (wreal)0.5 * rdnw[k] * ((ww_d[fp] + ww_d[f]) * dhi
                                              + (ww_pp[fp] + ww_pp[f]) * dhi_d);
        rhs[k] -= dtau * (fnm[k] * wd_hi + fnp[k] * wd_lo);
        rhsd[k] -= dtau * (fnm[k] * wd_hi_d + fnp[k] * wd_lo_d);
        wd_lo = wd_hi; wd_lo_d = wd_hi_d;
    }
    for (int k = 1; k <= nz; ++k) {
        wreal bf = c1f[k] * mut + c2f[k];
        rhsd[k] = phld[k] + msf * rhsd[k] / bf
                - msf * rhs[k] * c1f[k] * mut_d / (bf * bf);
        rhs[k] = phl[k] + msf * rhs[k] / bf;
    }
    if (top_lid) { rhs[nz] = 0; rhsd[nz] = 0; }

    // Bottom kinematic condition.
    WBC b = w_bc(j, i, ny, nx, boundary_x, boundary_y);
    wl[0] = w_bottom(u_pp, v_pp, ht, c, b, lu, lv, cf1, cf2, cf3, rdx, rdy, msf);
    wld[0] = w_bottom(u_d, v_d, ht, c, b, lu, lv, cf1, cf2, cf3, rdx, rdy, msf);

    // Forcing rows.
    wreal t2_dn = t2ave_(th_pp[c], th_pp_old[c], muave, c1h[0], c2h[0], muts,
                         thp[c], e);
    wreal t2_dn_d;
    {
        wreal dn_ = (c1h[0] * muts + c2h[0]) * (WT0 + thp[c]);
        wreal n_d = (wreal)0.5 * (((wreal)1.0 + e) * th_d[c] + ((wreal)1.0 - e) * thold_d[c])
                  + c1h[0] * WT0 * muave_d;
        wreal dn_d = c1h[0] * muts_d * (WT0 + thp[c]) + (c1h[0] * muts + c2h[0]) * thp_d[c];
        t2_dn_d = (n_d - t2_dn * dn_d) / dn_;
    }
    for (int k = 1; k < nz; ++k) {
        size_t h = (size_t)k * st + c, hm = h - st;
        wreal t2_up = t2ave_(th_pp[h], th_pp_old[h], muave, c1h[k], c2h[k], muts,
                             thp[h], e);
        wreal dn_ = (c1h[k] * muts + c2h[k]) * (WT0 + thp[h]);
        wreal n_d = (wreal)0.5 * (((wreal)1.0 + e) * th_d[h] + ((wreal)1.0 - e) * thold_d[h])
                  + c1h[k] * WT0 * muave_d;
        wreal dn_d = c1h[k] * muts_d * (WT0 + thp[h]) + (c1h[k] * muts + c2h[k]) * thp_d[h];
        wreal t2_up_d = (n_d - t2_up * dn_d) / dn_;

        wreal dph_up = ((wreal)1.0 + e) * (rhs[k + 1] - rhs[k])
                     + ((wreal)1.0 - e) * (phl[k + 1] - phl[k]);
        wreal dph_dn = ((wreal)1.0 + e) * (rhs[k] - rhs[k - 1])
                     + ((wreal)1.0 - e) * (phl[k] - phl[k - 1]);
        wreal dph_up_d = ((wreal)1.0 + e) * (rhsd[k + 1] - rhsd[k])
                       + ((wreal)1.0 - e) * (phld[k + 1] - phld[k]);
        wreal dph_dn_d = ((wreal)1.0 + e) * (rhsd[k] - rhsd[k - 1])
                       + ((wreal)1.0 - e) * (phld[k] - phld[k - 1]);
        wreal bh = c1h[k] * mut + c2h[k], bhm = c1h[k - 1] * mut + c2h[k - 1];
        wreal eu = c2a[h] * rdnw[k] / bh, ed = c2a[hm] * rdnw[k - 1] / bhm;
        wreal eu_d = rdnw[k] * (c2a_d[h] / bh - c2a[h] * c1h[k] * mut_d / (bh * bh));
        wreal ed_d = rdnw[k - 1] * (c2a_d[hm] / bhm
                                    - c2a[hm] * c1h[k - 1] * mut_d / (bhm * bhm));
        wreal q = c2a[h] * rdnw[k] / bh * dph_up - c2a[hm] * rdnw[k - 1] / bhm * dph_dn;
        wreal q_d = eu_d * dph_up + eu * dph_up_d - ed_d * dph_dn - ed * dph_dn_d;
        wreal pk = (wreal)0.5 * dtau * WG * rdn[k];
        wreal r_d = rdn[k] * ((c2a_d[h] * alt[h] + c2a[h] * alt_d[h]) * t2_up
                              + c2a[h] * alt[h] * t2_up_d
                              - (c2a_d[hm] * alt[hm] + c2a[hm] * alt_d[hm]) * t2_dn
                              - c2a[hm] * alt[hm] * t2_dn_d)
                  - c1f[k] * muave_d;
        size_t f = h;
        wreal fd;
        if (moist_cq) {
            wl[k] = wl[k] + dtau * rw_t[f]
                  + msf_i * cqw[f] * ((wreal)0.5 * dtau * WG * rdn[k]
                    * (c2a[h] * rdnw[k] / (c1h[k] * mut + c2h[k]) * dph_up
                       - c2a[hm] * rdnw[k - 1] / (c1h[k - 1] * mut + c2h[k - 1]) * dph_dn))
                  + dtau * WG * msf_i * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                                  - c2a[hm] * alt[hm] * t2_dn)
                                         - c1f[k] * muave);
            fd = msf_i * (cqw_d[f] * pk * q + cqw[f] * pk * q_d);
        } else {
            wl[k] = wl[k] + dtau * rw_t[f]
                  + msf_i * ((wreal)0.5 * dtau * WG * rdn[k]
                    * (c2a[h] * rdnw[k] / (c1h[k] * mut + c2h[k]) * dph_up
                       - c2a[hm] * rdnw[k - 1] / (c1h[k - 1] * mut + c2h[k - 1]) * dph_dn))
                  + dtau * WG * msf_i * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                                  - c2a[hm] * alt[hm] * t2_dn)
                                         - c1f[k] * muave);
            fd = msf_i * pk * q_d;
        }
        wld[k] = wld[k] + dtau * rw_d[f] + fd + dtau * WG * msf_i * r_d;
        t2_dn = t2_up; t2_dn_d = t2_up_d;
    }
    if (top_lid) {
        wl[nz] = 0; wld[nz] = 0;
    } else {
        int h1 = nz - 1;
        size_t h = (size_t)h1 * st + c, f = (size_t)nz * st + c;
        wreal dph_dn = ((wreal)1.0 + e) * (rhs[nz] - rhs[nz - 1])
                     + ((wreal)1.0 - e) * (phl[nz] - phl[nz - 1]);
        wreal dph_dn_d = ((wreal)1.0 + e) * (rhsd[nz] - rhsd[nz - 1])
                       + ((wreal)1.0 - e) * (phld[nz] - phld[nz - 1]);
        wreal bh = c1h[h1] * mut + c2h[h1];
        wreal kk = -dtau * WG * rdnw[h1] * rdnw[h1];
        wreal t_d = kk * ((c2a_d[h] / bh - c2a[h] * c1h[h1] * mut_d / (bh * bh)) * dph_dn
                          + c2a[h] / bh * dph_dn_d)
                  - dtau * WG * ((wreal)2.0 * rdnw[h1]
                                 * ((c2a_d[h] * alt[h] + c2a[h] * alt_d[h]) * t2_dn
                                    + c2a[h] * alt[h] * t2_dn_d)
                                 + c1f[nz] * muave_d);
        wl[nz] = wl[nz] + dtau * rw_t[f]
               + msf_i * ((-(wreal)0.5 * dtau * WG) / (c1h[h1] * mut + c2h[h1])
                  * (rdnw[h1] * rdnw[h1]) * (wreal)2.0 * c2a[h] * dph_dn
                  - dtau * WG * ((wreal)2.0 * rdnw[h1] * c2a[h] * alt[h] * t2_dn
                                 + c1f[nz] * muave));
        wld[nz] = wld[nz] + dtau * rw_d[f] + msf_i * t_d;
    }

    // Thomas solve.
    for (int k = 1; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        wreal x = wl[k] - a[f] * wl[k - 1];
        wld[k] = (wld[k] - a_d[f] * wl[k - 1] - a[f] * wld[k - 1]) * alpha[f]
               + x * alpha_d[f];
        wl[k] = (wl[k] - a[f] * wl[k - 1]) * alpha[f];
    }
    for (int k = nz - 1; k >= 1; --k) {
        size_t f = (size_t)k * st + c;
        wld[k] = wld[k] - gam_d[f] * wl[k + 1] - gam[f] * wld[k + 1];
        wl[k] -= gam[f] * wl[k + 1];
    }

    // damp_opt = 3 w damper (WRFPLUS g_advance_w, with woof's hybrid
    // c1f mut + c2f where WRFPLUS has mut).
    if (dampmag > (wreal)0.0) {
        wreal htop = (phb[(size_t)nz * bstr + boff] + php[(size_t)nz * st + c]) / WG;
        wreal hbot = htop - zdamp;
        wreal hbot_d = php_d[(size_t)nz * st + c] / WG;
        for (int k = 1; k <= nz; ++k) {
            size_t f = (size_t)k * st + c;
            wreal hk = (phb[(size_t)k * bstr + boff] + php[f]) / WG;
            if (hk >= hbot) {
                wreal angle = WPI2 * (hk - hbot) / zdamp;
                wreal sn = wsin(angle);
                wreal ang_d = WPI2 * (php_d[f] / WG - hbot_d) / zdamp;
                wreal dw = dampmag * sn * sn;
                wreal dw_d = (wreal)2.0 * dampmag * sn * wcos(angle) * ang_d;
                wreal cfm = c1f[k] * mut + c2f[k];
                wreal num = wl[k] - dw * cfm * w_ref[f];
                wreal num_d = wld[k] - (dw_d * cfm + dw * c1f[k] * mut_d) * w_ref[f]
                            - dw * cfm * wref_d[f];
                wl[k] = num / ((wreal)1.0 + dw);
                wld[k] = (num_d - wl[k] * dw_d) / ((wreal)1.0 + dw);
            }
        }
    }

    // phi'' update.
    for (int k = 1; k <= nz; ++k) {
        wreal bs = c1f[k] * muts + c2f[k];
        wreal cw = msf * (wreal)0.5 * dtau * WG * ((wreal)1.0 + e);
        phld[k] = rhsd[k] + cw * (wld[k] / bs - wl[k] * c1f[k] * muts_d / (bs * bs));
        phl[k] = rhs[k] + msf * (wreal)0.5 * dtau * WG * ((wreal)1.0 + e) * wl[k]
                        / (c1f[k] * muts + c2f[k]);
    }

    // calc_p_rho.
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        wreal chm = c1h[k] * (mub2d[c] + mup[c] + mu_pp[c]) + c2h[k];
        wreal chm_d = c1h[k] * muts_d;
        wreal th_ref = thp[h];
        wreal s = alt[h] * (c1h[k] * mu_pp[c]) + rdnw[k] * (phl[k + 1] - phl[k]);
        wreal s_d = alt_d[h] * c1h[k] * mu_pp[c] + alt[h] * c1h[k] * mupp_d[c]
                  + rdnw[k] * (phld[k + 1] - phld[k]);
        wreal al = (wreal)-1.0 / chm * s;
        wreal ald = (-s_d - al * chm_d) / chm;
        wreal nv = alt[h] * (th_pp[h] - (c1h[k] * mu_pp[c]) * th_ref);
        wreal dv = chm * (WT0 + th_ref);
        wreal v = nv / dv;
        wreal nv_d = alt_d[h] * (th_pp[h] - c1h[k] * mu_pp[c] * th_ref)
                   + alt[h] * (th_d[h] - c1h[k] * (mupp_d[c] * th_ref + mu_pp[c] * thp_d[h]));
        wreal dv_d = chm_d * (WT0 + th_ref) + chm * thp_d[h];
        wreal v_dd = (nv_d - v * dv_d) / dv;
        if (WRFPLUS_TL) p_d[h] = c2a_d[h] * (v - al) + c2a[h] * (v_dd - ald);
        if (WRFPLUS_TL) al_d[h] = ald;
        if (nl) {
            al_pp[h] = al;
            p_pp[h] = c2a[h] * (alt[h] * (th_pp[h] - (c1h[k] * mu_pp[c]) * th_ref)
                                / (chm * (WT0 + th_ref)) - al);
        }
    }
    for (int k = 0; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        if (WRFPLUS_TL) w_d[f] = wld[k];
        if (WRFPLUS_TL && k >= 1) ph_d[f] = phld[k];
        if (nl) {
            w_pp[f] = wl[k];
            if (k >= 1) ph_pp[f] = phl[k];
        }
    }
}

// Adjoint, column pass: WRFPLUS a_calc_p_rho then a_advance_w.  On entry
// w_a, ph_a, p_a, al_a hold the adjoints of the outputs; on exit w_a and
// ph_a hold those of the inputs and p_a, al_a are zero (interior columns;
// the frame is untouched, as in the forward).  The other active inputs'
// adjoints are added.  The bottom condition's adjoint is left in aw0 for
// advance_w_ad_bc.
extern "C" __global__
void advance_w_ad(wreal* __restrict__ w_a, wreal* __restrict__ ph_a,
                  wreal* __restrict__ p_a, wreal* __restrict__ al_a,
                  const wreal* __restrict__ w_pp, const wreal* __restrict__ ph_pp,
                  W_TRAJ_PARAMS,
                  wreal* __restrict__ rw_a, wreal* __restrict__ rph_a,
                  wreal* __restrict__ ww_a, wreal* __restrict__ mupp_a,
                  wreal* __restrict__ mupold_a, wreal* __restrict__ th_a,
                  wreal* __restrict__ thold_a, wreal* __restrict__ thp_a,
                  wreal* __restrict__ php_a, wreal* __restrict__ alt_a,
                  wreal* __restrict__ c2a_a, wreal* __restrict__ a_a,
                  wreal* __restrict__ alpha_a, wreal* __restrict__ gam_a,
                  wreal* __restrict__ mup_a, wreal* __restrict__ cqw_a,
                  wreal* __restrict__ wref_a,
                  wreal* __restrict__ aw0, W_GEOM_PARAMS)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx || nz + 1 > WV_LEV) return;
    int j = c / nx, i = c - j * nx;
    if (i < spec_zone || i >= nx - spec_zone
        || j < spec_zone || j >= ny - spec_zone) {
        aw0[c] = 0;
        return;
    }
    size_t st = (size_t)ny * nx;
    size_t bstr = base3d ? st : 1;
    size_t boff = base3d ? (size_t)c : 0;
    size_t lu = (size_t)ny * (nx + 1), lv = (size_t)(ny + 1) * nx;
    wreal e = epssm;
    // WRF advance_w's msfty and msft_inv = 1/msfty (exact 1 without msf).
    wreal msf = has_msf ? msft[c] : (wreal)1.0;
    wreal msf_i = (wreal)1.0 / msf;

    // ---- Forward recompute (base state only). ----
    wreal mut = mub2d[c] + mup[c];
    wreal muts = mut + mu_pp[c];
    wreal muave = (wreal)0.5 * (((wreal)1.0 + e) * mu_pp[c]
                              + ((wreal)1.0 - e) * mu_pp_old[c]);
    wreal phl[WV_LEV], rhs2[WV_LEV], rhs3[WV_LEV], t2[WV_LEV];
    wreal w5[WV_LEV], w6[WV_LEV], w7[WV_LEV], w8[WV_LEV], pho[WV_LEV];
    for (int k = 0; k <= nz; ++k) phl[k] = ph_pp[(size_t)k * st + c];
    rhs2[0] = 0;
    for (int k = 1; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        rhs2[k] = dtau * (rph_t[f] + (wreal)0.5 * WG * ((wreal)1.0 - e) * w_pp[f]);
    }
    {
        wreal dlo = (php[st + c] - php[c] + phb[bstr + boff] - phb[boff]);
        wreal wd_lo = (wreal)0.5 * (ww_pp[st + c] + ww_pp[c]) * rdnw[0] * dlo;
        for (int k = 1; k < nz; ++k) {
            size_t f = (size_t)k * st + c, fp = f + st;
            wreal dhi = (php[fp] - php[f]
                         + phb[(size_t)(k + 1) * bstr + boff] - phb[(size_t)k * bstr + boff]);
            wreal wd_hi = (wreal)0.5 * (ww_pp[fp] + ww_pp[f]) * rdnw[k] * dhi;
            rhs2[k] -= dtau * (fnm[k] * wd_hi + fnp[k] * wd_lo);
            wd_lo = wd_hi;
        }
    }
    rhs3[0] = 0;
    for (int k = 1; k <= nz; ++k)
        rhs3[k] = phl[k] + msf * rhs2[k] / (c1f[k] * mut + c2f[k]);
    if (top_lid) rhs3[nz] = 0;
    WBC b = w_bc(j, i, ny, nx, boundary_x, boundary_y);
    w5[0] = w_bottom(u_pp, v_pp, ht, c, b, lu, lv, cf1, cf2, cf3, rdx, rdy, msf);
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        t2[k] = t2ave_(th_pp[h], th_pp_old[h], muave, c1h[k], c2h[k], muts, thp[h], e);
    }
    for (int k = 1; k < nz; ++k) {
        size_t h = (size_t)k * st + c, hm = h - st;
        wreal dph_up = ((wreal)1.0 + e) * (rhs3[k + 1] - rhs3[k])
                     + ((wreal)1.0 - e) * (phl[k + 1] - phl[k]);
        wreal dph_dn = ((wreal)1.0 + e) * (rhs3[k] - rhs3[k - 1])
                     + ((wreal)1.0 - e) * (phl[k] - phl[k - 1]);
        wreal pq = (wreal)0.5 * dtau * WG * rdn[k]
                 * (c2a[h] * rdnw[k] / (c1h[k] * mut + c2h[k]) * dph_up
                    - c2a[hm] * rdnw[k - 1] / (c1h[k - 1] * mut + c2h[k - 1]) * dph_dn);
        w5[k] = w_pp[h] + dtau * rw_t[h] + (moist_cq ? msf_i * cqw[h] * pq : msf_i * pq)
              + dtau * WG * msf_i
                * (rdn[k] * (c2a[h] * alt[h] * t2[k] - c2a[hm] * alt[hm] * t2[k - 1])
                   - c1f[k] * muave);
    }
    if (top_lid) {
        w5[nz] = 0;
    } else {
        int h1 = nz - 1;
        size_t h = (size_t)h1 * st + c, f = (size_t)nz * st + c;
        wreal dph_dn = ((wreal)1.0 + e) * (rhs3[nz] - rhs3[nz - 1])
                     + ((wreal)1.0 - e) * (phl[nz] - phl[nz - 1]);
        w5[nz] = w_pp[f] + dtau * rw_t[f]
               + msf_i * ((-(wreal)0.5 * dtau * WG) / (c1h[h1] * mut + c2h[h1])
                  * (rdnw[h1] * rdnw[h1]) * (wreal)2.0 * c2a[h] * dph_dn
                  - dtau * WG * ((wreal)2.0 * rdnw[h1] * c2a[h] * alt[h] * t2[h1]
                                 + c1f[nz] * muave));
    }
    w6[0] = w5[0];
    for (int k = 1; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        w6[k] = (w5[k] - a[f] * w6[k - 1]) * alpha[f];
    }
    w7[nz] = w6[nz]; w7[0] = w6[0];
    for (int k = nz - 1; k >= 1; --k) {
        size_t f = (size_t)k * st + c;
        w7[k] = w6[k] - gam[f] * w7[k + 1];
    }
    // The damper (w7 -> w8; w8 = w7 where it does not act).
    wreal htop = 0, hbot = 0;
    if (dampmag > (wreal)0.0) {
        htop = (phb[(size_t)nz * bstr + boff] + php[(size_t)nz * st + c]) / WG;
        hbot = htop - zdamp;
    }
    for (int k = 0; k <= nz; ++k) {
        w8[k] = w7[k];
        if (dampmag > (wreal)0.0 && k >= 1) {
            size_t f = (size_t)k * st + c;
            wreal hk = (phb[(size_t)k * bstr + boff] + php[f]) / WG;
            if (hk >= hbot) {
                wreal sn = wsin(WPI2 * (hk - hbot) / zdamp);
                wreal dw = dampmag * sn * sn;
                w8[k] = (w7[k] - dw * (c1f[k] * mut + c2f[k]) * w_ref[f])
                      / ((wreal)1.0 + dw);
            }
        }
    }
    pho[0] = phl[0];
    for (int k = 1; k <= nz; ++k)
        pho[k] = rhs3[k] + msf * (wreal)0.5 * dtau * WG * ((wreal)1.0 + e) * w8[k]
                         / (c1f[k] * muts + c2f[k]);

    // ---- Reverse sweep. ----
    wreal amut = 0, amuts = 0, amuave = 0, amupp = 0;
    // Adjoints of the outputs.  aw holds w7 -> w6 -> w5 in turn; aph
    // accumulates the adjoint of the input ph.
    wreal aw[WV_LEV], apho[WV_LEV], aph[WV_LEV], arhs[WV_LEV], at2[WV_LEV];
    for (int k = 0; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        aw[k] = w_a[f];
        apho[k] = ph_a[f];
        aph[k] = 0; arhs[k] = 0;
    }
    for (int k = 0; k < nz; ++k) at2[k] = 0;

    // calc_p_rho.
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        wreal chm = c1h[k] * muts + c2h[k];
        wreal th_ref = thp[h];
        wreal s = alt[h] * (c1h[k] * mu_pp[c]) + rdnw[k] * (pho[k + 1] - pho[k]);
        wreal al = (wreal)-1.0 / chm * s;
        wreal nv = alt[h] * (th_pp[h] - c1h[k] * mu_pp[c] * th_ref);
        wreal dv = chm * (WT0 + th_ref);
        wreal v = nv / dv;
        wreal ap = p_a[h];
        wreal aal = al_a[h] - ap * c2a[h];
        c2a_a[h] += ap * (v - al);
        wreal av = ap * c2a[h];
        wreal anv = av / dv;
        wreal adv = -av * v / dv;
        wreal achm = adv * (WT0 + th_ref);
        thp_a[h] += adv * chm - anv * alt[h] * c1h[k] * mu_pp[c];
        alt_a[h] += anv * (th_pp[h] - c1h[k] * mu_pp[c] * th_ref);
        th_a[h] += anv * alt[h];
        amupp += -anv * alt[h] * c1h[k] * th_ref;
        wreal as = -aal / chm;
        achm += -aal * al / chm;
        alt_a[h] += as * c1h[k] * mu_pp[c];
        amupp += as * alt[h] * c1h[k];
        apho[k + 1] += as * rdnw[k];
        apho[k] -= as * rdnw[k];
        amuts += achm * c1h[k];
        p_a[h] = 0;
        al_a[h] = 0;
    }
    // phi'' update.
    aph[0] += apho[0];
    for (int k = 1; k <= nz; ++k) {
        wreal bs = c1f[k] * muts + c2f[k];
        wreal cw = msf * (wreal)0.5 * dtau * WG * ((wreal)1.0 + e);
        arhs[k] += apho[k];
        aw[k] += apho[k] * cw / bs;
        amuts += -apho[k] * cw * w8[k] * c1f[k] / (bs * bs);
    }
    // The damper (aw: w8 -> w7).
    if (dampmag > (wreal)0.0) {
        for (int k = 1; k <= nz; ++k) {
            size_t f = (size_t)k * st + c;
            wreal hk = (phb[(size_t)k * bstr + boff] + php[f]) / WG;
            if (!(hk >= hbot)) continue;
            wreal angle = WPI2 * (hk - hbot) / zdamp;
            wreal sn = wsin(angle);
            wreal dw = dampmag * sn * sn;
            wreal cfm = c1f[k] * mut + c2f[k];
            wreal anum = aw[k] / ((wreal)1.0 + dw);
            wreal adw = -anum * w8[k] - anum * cfm * w_ref[f];
            amut += -anum * dw * w_ref[f] * c1f[k];
            wref_a[f] += -anum * dw * cfm;
            aw[k] = anum;
            wreal ah = (wreal)2.0 * dampmag * sn * wcos(angle) * adw * WPI2 / zdamp / WG;
            php_a[f] += ah;
            php_a[(size_t)nz * st + c] -= ah;
        }
    }
    // Back substitution (aw: w7 -> w6).
    for (int k = 1; k <= nz - 1; ++k) {
        size_t f = (size_t)k * st + c;
        gam_a[f] += -aw[k] * w7[k + 1];
        aw[k + 1] += -aw[k] * gam[f];
    }
    // Forward elimination (aw: w6 -> w5).
    for (int k = nz; k >= 1; --k) {
        size_t f = (size_t)k * st + c;
        wreal x = w5[k] - a[f] * w6[k - 1];
        wreal ax = aw[k] * alpha[f];
        alpha_a[f] += aw[k] * x;
        aw[k] = ax;
        a_a[f] += -ax * w6[k - 1];
        aw[k - 1] += -ax * a[f];
    }
    // Top row.
    wreal awin[WV_LEV];
    for (int k = 0; k <= nz; ++k) awin[k] = 0;
    if (!top_lid) {
        int h1 = nz - 1;
        size_t h = (size_t)h1 * st + c, f = (size_t)nz * st + c;
        wreal dph_dn = ((wreal)1.0 + e) * (rhs3[nz] - rhs3[nz - 1])
                     + ((wreal)1.0 - e) * (phl[nz] - phl[nz - 1]);
        wreal bh = c1h[h1] * mut + c2h[h1];
        wreal kk = -dtau * WG * rdnw[h1] * rdnw[h1];
        awin[nz] += aw[nz];
        rw_a[f] += dtau * aw[nz];
        wreal at = msf_i * aw[nz];        // the msft_inv-scaled terms
        c2a_a[h] += at * kk * dph_dn / bh
                  - at * dtau * WG * (wreal)2.0 * rdnw[h1] * alt[h] * t2[h1];
        amut += -at * kk * c2a[h] * dph_dn / (bh * bh) * c1h[h1];
        wreal adph = at * kk * c2a[h] / bh;
        alt_a[h] += -at * dtau * WG * (wreal)2.0 * rdnw[h1] * c2a[h] * t2[h1];
        at2[h1] += -at * dtau * WG * (wreal)2.0 * rdnw[h1] * c2a[h] * alt[h];
        amuave += -at * dtau * WG * c1f[nz];
        arhs[nz] += ((wreal)1.0 + e) * adph;
        arhs[nz - 1] -= ((wreal)1.0 + e) * adph;
        aph[nz] += ((wreal)1.0 - e) * adph;
        aph[nz - 1] -= ((wreal)1.0 - e) * adph;
    }
    // Forcing rows.
    for (int k = 1; k < nz; ++k) {
        size_t h = (size_t)k * st + c, hm = h - st;
        wreal dph_up = ((wreal)1.0 + e) * (rhs3[k + 1] - rhs3[k])
                     + ((wreal)1.0 - e) * (phl[k + 1] - phl[k]);
        wreal dph_dn = ((wreal)1.0 + e) * (rhs3[k] - rhs3[k - 1])
                     + ((wreal)1.0 - e) * (phl[k] - phl[k - 1]);
        wreal bh = c1h[k] * mut + c2h[k], bhm = c1h[k - 1] * mut + c2h[k - 1];
        wreal eu = c2a[h] * rdnw[k] / bh, ed = c2a[hm] * rdnw[k - 1] / bhm;
        wreal q = eu * dph_up - ed * dph_dn;
        wreal pk = (wreal)0.5 * dtau * WG * rdn[k];
        wreal a5 = aw[k];
        awin[k] += a5;
        rw_a[h] += dtau * a5;
        wreal aq;
        if (moist_cq) {
            cqw_a[h] += msf_i * a5 * pk * q;
            aq = msf_i * a5 * cqw[h] * pk;
        } else {
            aq = msf_i * a5 * pk;
        }
        wreal ar = dtau * WG * msf_i * a5;
        wreal aeu = aq * dph_up, aed = -aq * dph_dn;
        wreal adu = aq * eu, add = -aq * ed;
        arhs[k + 1] += ((wreal)1.0 + e) * adu;
        arhs[k] -= ((wreal)1.0 + e) * adu;
        aph[k + 1] += ((wreal)1.0 - e) * adu;
        aph[k] -= ((wreal)1.0 - e) * adu;
        arhs[k] += ((wreal)1.0 + e) * add;
        arhs[k - 1] -= ((wreal)1.0 + e) * add;
        aph[k] += ((wreal)1.0 - e) * add;
        aph[k - 1] -= ((wreal)1.0 - e) * add;
        c2a_a[h] += aeu * rdnw[k] / bh;
        amut += -aeu * rdnw[k] * c2a[h] / (bh * bh) * c1h[k];
        c2a_a[hm] += aed * rdnw[k - 1] / bhm;
        amut += -aed * rdnw[k - 1] * c2a[hm] / (bhm * bhm) * c1h[k - 1];
        wreal rr = ar * rdn[k];
        c2a_a[h] += rr * alt[h] * t2[k];
        alt_a[h] += rr * c2a[h] * t2[k];
        at2[k] += rr * c2a[h] * alt[h];
        c2a_a[hm] += -rr * alt[hm] * t2[k - 1];
        alt_a[hm] += -rr * c2a[hm] * t2[k - 1];
        at2[k - 1] += -rr * c2a[hm] * alt[hm];
        amuave += -ar * c1f[k];
    }
    // t_2ave.
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        wreal dn_ = (c1h[k] * muts + c2h[k]) * (WT0 + thp[h]);
        wreal an = at2[k] / dn_;
        wreal adn = -at2[k] * t2[k] / dn_;
        th_a[h] += (wreal)0.5 * ((wreal)1.0 + e) * an;
        thold_a[h] += (wreal)0.5 * ((wreal)1.0 - e) * an;
        amuave += c1h[k] * WT0 * an;
        amuts += adn * c1h[k] * (WT0 + thp[h]);
        thp_a[h] += adn * (c1h[k] * muts + c2h[k]);
    }
    // Bottom condition.
    aw0[c] = msf * aw[0];
    // rhs3 = ph + rhs2 / bf  (rhs3[nz] = 0 under the lid).
    for (int k = 1; k <= nz; ++k) {
        if (top_lid && k == nz) { arhs[k] = 0; continue; }
        wreal bf = c1f[k] * mut + c2f[k];
        aph[k] += arhs[k];
        amut += -arhs[k] * msf * rhs2[k] * c1f[k] / (bf * bf);
        arhs[k] = msf * arhs[k] / bf;
    }
    // rhs2 = rhs1 - dtau*(fnm*wd[k] + fnp*wd[k-1]); wd from ww and php.
    for (int k = 1; k < nz; ++k) {
        wreal r = arhs[k];
        for (int m = k - 1; m <= k; ++m) {
            wreal awd = -dtau * (m == k ? fnm[k] : fnp[k]) * r;
            size_t f = (size_t)m * st + c, fp = f + st;
            wreal dm = (php[fp] - php[f]
                        + phb[(size_t)(m + 1) * bstr + boff] - phb[(size_t)m * bstr + boff]);
            wreal g1 = awd * (wreal)0.5 * rdnw[m];
            ww_a[fp] += g1 * dm;
            ww_a[f] += g1 * dm;
            php_a[fp] += g1 * (ww_pp[fp] + ww_pp[f]);
            php_a[f] -= g1 * (ww_pp[fp] + ww_pp[f]);
        }
    }
    // rhs1 = dtau*(rph + 0.5*g*(1-e)*w).
    for (int k = 1; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        rph_a[f] += dtau * arhs[k];
        awin[k] += dtau * (wreal)0.5 * WG * ((wreal)1.0 - e) * arhs[k];
    }
    // Column scalars.
    mup_a[c] += amut + amuts;
    mupp_a[c] += amuts + (wreal)0.5 * ((wreal)1.0 + e) * amuave + amupp;
    mupold_a[c] += (wreal)0.5 * ((wreal)1.0 - e) * amuave;
    for (int k = 0; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        w_a[f] = awin[k];
        ph_a[f] = aph[k];
    }
}

// Adjoint of the bottom condition: one thread per (level 0..2, face).  A u
// face is the east face (ue) of column i-1 and the west face (uw) of
// column i; a v face the north face of row j-1 and the south of row j.
extern "C" __global__
void advance_w_ad_bc(const wreal* __restrict__ aw0, const wreal* __restrict__ ht,
                     wreal* __restrict__ u_a, wreal* __restrict__ v_a,
                     W_GEOM_PARAMS)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= 3 * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf, i = r - j * nxf;
    wreal cf = k == 0 ? cf1 : k == 1 ? cf2 : cf3;
    if (j < ny) {
        wreal s = 0;
        int ic = i - 1;
        if (ic >= spec_zone && ic < nx - spec_zone
            && j >= spec_zone && j < ny - spec_zone) {
            WBC b = w_bc(j, ic, ny, nx, boundary_x, boundary_y);
            size_t cc = (size_t)j * nx + ic;
            s += (wreal)0.5 * rdx * (ht[b.cip] - ht[cc]) * cf * aw0[cc];
        }
        if (i >= spec_zone && i < nx - spec_zone
            && j >= spec_zone && j < ny - spec_zone) {
            WBC b = w_bc(j, i, ny, nx, boundary_x, boundary_y);
            size_t cc = (size_t)j * nx + i;
            s += (wreal)0.5 * rdx * (ht[cc] - ht[b.cim]) * cf * aw0[cc];
        }
        u_a[W3S(k, j, i, ny, nxf)] += s;
    }
    if (i < nx) {
        wreal s = 0;
        int jc = j - 1;
        if (jc >= spec_zone && jc < ny - spec_zone
            && i >= spec_zone && i < nx - spec_zone) {
            WBC b = w_bc(jc, i, ny, nx, boundary_x, boundary_y);
            size_t cc = (size_t)jc * nx + i;
            s += (wreal)0.5 * rdy * (ht[b.cjp] - ht[cc]) * cf * aw0[cc];
        }
        if (j < ny && j >= spec_zone && j < ny - spec_zone
            && i >= spec_zone && i < nx - spec_zone) {
            WBC b = w_bc(j, i, ny, nx, boundary_x, boundary_y);
            size_t cc = (size_t)j * nx + i;
            s += (wreal)0.5 * rdy * (ht[cc] - ht[b.cjm]) * cf * aw0[cc];
        }
        v_a[W3S(k, j, i, nyf, nx)] += s;
    }
}
