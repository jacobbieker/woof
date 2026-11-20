// Tangent-linear and adjoint of woof's strict advance_mu_th (the
// GPUWM_WRF_EXACT wrf_advance_mu_theta in woof/core/kernels/acoustic.cu:
// WRF advance_mu_t) and of advance_exact_frame_mu_t (solve_em's
// spec_bdyupdate of t_2, mu_2 and muts on the specified frame).  The
// oracle is WRFPLUS 4.7.1 g_/a_advance_mu_t and g_/a_spec_bdyupdate
// (wrftladj/module_small_step_em_{tl,ad}.F, module_bc_{tl,ad}.F); see
// docs/dev/P3A-WRF-DRYCORE-ADJOINT.md.
//
// wreal is binary64 unless the unit is built with -DWRFPLUS_REAL=float.
// The base-state statements keep the strict primal's operation order, so
// the float instantiation of the TL's base state gives the primal's words.
// The primal divides with __fdiv_rn and this unit with plain '/': the two
// agree only while the unit is compiled with IEEE division (NVRTC's
// default -prec-div=true, never -use_fast_math) and --fmad=false
// (acoustic.py _module); the strict-primal bitwise test catches a change.
//
// Layout (k, j, i).  mu_pp, mu_old, mudf, muts, mup, rmu_t (ny, nx);
// th_pp, th_old, p_old, thp, rth_t, p_pp (nz, ny, nx); ww_pp, ww_ref at
// w levels (nz+1, ny, nx); u_pp, u (nz, ny, nx+1); v_pp, v (nz, ny+1, nx).
//
// State (in/out): mu_pp, th_pp; outputs mu_old, th_old, p_old (the
// history copies), mudf (when write_mudf) and ww_pp (solved columns
// only, as in the primal); muts (frame kernel).  Active inputs: u_pp,
// v_pp, u, v, mup, rmu_t, thp (theta), rth_t, ww_ref, p_pp.  Passive:
// dnw, rdnw, fnm, fnp, c1h, c2h, mub2d, msft, msfu, msfv.
//
// The adjoint is a column pass (one thread per column: recompute, then
// the reverse sweep) that leaves each column's layer-divergence adjoint
// D = a_dvdxi*msft^2 and horizontal theta-flux adjoint H = a_hx = a_hy in
// scratch, then a gather pass in which each column adds the face terms it
// owns into u_pp/u/v_pp/v and gathers its theta and mup adjoints from the
// faces it touches.  Every output word has one writer: no atomics, and
// the result does not depend on thread order.

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

#define MI3(k, j, i, nyy, nxx) \
    ((size_t)(k) * (size_t)(nyy) * (size_t)(nxx) + (size_t)(j) * (size_t)(nxx) + (size_t)(i))

#define MT_PASSIVE                                                          \
    const wreal* __restrict__ dnw, const wreal* __restrict__ rdnw,         \
    const wreal* __restrict__ fnm, const wreal* __restrict__ fnp,          \
    const wreal* __restrict__ c1h, const wreal* __restrict__ c2h,          \
    const wreal* __restrict__ mub2d, const wreal* __restrict__ msft,       \
    const wreal* __restrict__ msfu, const wreal* __restrict__ msfv

#define MT_GEOM int nz, int ny, int nx, int open_x, int open_y,              \
    int spec_zone, int mapped, int write_mudf, wreal rdx, wreal rdy,       \
    wreal dtau

#define MT_TRAJ(s)                                                          \
    const wreal* __restrict__ u_pp##s, const wreal* __restrict__ v_pp##s,  \
    const wreal* __restrict__ u##s, const wreal* __restrict__ v##s,        \
    const wreal* __restrict__ mup##s, const wreal* __restrict__ rmu_t##s,  \
    const wreal* __restrict__ thp##s, const wreal* __restrict__ rth_t##s,  \
    const wreal* __restrict__ ww_ref##s, const wreal* __restrict__ p_pp##s

#define MT_TRAJ_ADJ                                                         \
    wreal* __restrict__ u_pp_a, wreal* __restrict__ v_pp_a,                \
    wreal* __restrict__ u_a, wreal* __restrict__ v_a,                      \
    wreal* __restrict__ mup_a, wreal* __restrict__ rmu_t_a,                \
    wreal* __restrict__ thp_a, wreal* __restrict__ rth_t_a,                \
    wreal* __restrict__ ww_ref_a, wreal* __restrict__ p_pp_a

#define MT_STATE(s)                                                         \
    wreal* __restrict__ mu_pp##s, wreal* __restrict__ th_pp##s,            \
    wreal* __restrict__ mu_old##s, wreal* __restrict__ th_old##s,          \
    wreal* __restrict__ p_old##s, wreal* __restrict__ mudf##s,             \
    wreal* __restrict__ ww_pp##s

static __device__ __forceinline__ bool mt_frame(int j, int i, int ny, int nx,
                                                int spec_zone)
{
    return i < spec_zone || i >= nx - spec_zone || j < spec_zone
        || j >= ny - spec_zone;
}

extern "C" __global__
void advance_mu_t_tl(MT_STATE(), MT_STATE(_d), MT_TRAJ(), MT_TRAJ(_d),
                     MT_PASSIVE, int nl, MT_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx || nz + 1 > WV_LEV) return;
    int j = c / nx, i = c - j * nx;
    size_t st = (size_t)ny * nx;
    if (nl) mu_old[c] = mu_pp[c];
    if (WRFPLUS_TL) mu_old_d[c] = mu_pp_d[c];
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        if (nl) { th_old[h] = th_pp[h]; p_old[h] = p_pp[h]; }
        if (WRFPLUS_TL) { th_old_d[h] = th_pp_d[h]; p_old_d[h] = p_pp_d[h]; }
    }
    if (mt_frame(j, i, ny, nx, spec_zone)) {
        if (write_mudf) { if (nl) mudf[c] = (wreal)0.0; if (WRFPLUS_TL) mudf_d[c] = 0; }
        return;
    }
    int ip = open_x ? min(i + 1, nx - 1) : (i + 1) % nx;
    int im = open_x ? max(i - 1, 0) : (i - 1 + nx) % nx;
    int jp = open_y ? min(j + 1, ny - 1) : (j + 1) % ny;
    int jm = open_y ? max(j - 1, 0) : (j - 1 + ny) % ny;
    size_t cw = (size_t)j * nx + im, ce = (size_t)j * nx + ip;
    size_t cs = (size_t)jm * nx + i, cn = (size_t)jp * nx + i;
    wreal muw = (wreal)0.5 * (mup[c] + mup[cw] + mub2d[c] + mub2d[cw]);
    wreal mue = (wreal)0.5 * (mup[ce] + mup[c] + mub2d[ce] + mub2d[c]);
    wreal mus = (wreal)0.5 * (mup[c] + mup[cs] + mub2d[c] + mub2d[cs]);
    wreal mun = (wreal)0.5 * (mup[cn] + mup[c] + mub2d[cn] + mub2d[c]);
    wreal muw_d = (wreal)0.5 * (mup_d[c] + mup_d[cw]);
    wreal mue_d = (wreal)0.5 * (mup_d[ce] + mup_d[c]);
    wreal mus_d = (wreal)0.5 * (mup_d[c] + mup_d[cs]);
    wreal mun_d = (wreal)0.5 * (mup_d[cn] + mup_d[c]);
    wreal mc = mapped ? msft[c] : (wreal)1.0;
    wreal uxw = mapped ? msfu[(size_t)j * (nx + 1) + i] : (wreal)1.0;
    wreal uxe = mapped ? msfu[(size_t)j * (nx + 1) + i + 1] : (wreal)1.0;
    wreal vys = mapped ? msfv[(size_t)j * nx + i] : (wreal)1.0;
    wreal vyn = mapped ? msfv[(size_t)(j + 1) * nx + i] : (wreal)1.0;

    wreal dv[WV_LEV], dvd[WV_LEV];
    wreal dmdt = 0, dmdt_d = 0;
    for (int k = 0; k < nz; ++k) {
        size_t uw = MI3(k, j, i, ny, nx + 1), ue = uw + 1;
        size_t vs = MI3(k, j, i, ny + 1, nx), vn = vs + nx;
        wreal vs_full = v_pp[vs] + (c1h[k] * mus + c2h[k]) * v[vs] * vys;
        wreal vn_full = v_pp[vn] + (c1h[k] * mun + c2h[k]) * v[vn] * vyn;
        wreal uw_full = u_pp[uw] + ((c1h[k] * muw + c2h[k]) * u[uw]) / uxw;
        wreal ue_full = u_pp[ue] + ((c1h[k] * mue + c2h[k]) * u[ue]) / uxe;
        dv[k] = mc * mc * (rdy * (vn_full - vs_full) + rdx * (ue_full - uw_full));
        wreal vs_d = v_pp_d[vs] + (c1h[k] * mus_d * v[vs]
                                   + (c1h[k] * mus + c2h[k]) * v_d[vs]) * vys;
        wreal vn_d = v_pp_d[vn] + (c1h[k] * mun_d * v[vn]
                                   + (c1h[k] * mun + c2h[k]) * v_d[vn]) * vyn;
        wreal uw_d = u_pp_d[uw] + (c1h[k] * muw_d * u[uw]
                                   + (c1h[k] * muw + c2h[k]) * u_d[uw]) / uxw;
        wreal ue_d = u_pp_d[ue] + (c1h[k] * mue_d * u[ue]
                                   + (c1h[k] * mue + c2h[k]) * u_d[ue]) / uxe;
        dvd[k] = mc * mc * (rdy * (vn_d - vs_d) + rdx * (ue_d - uw_d));
        dmdt = dmdt + dnw[k] * dv[k];
        dmdt_d = dmdt_d + dnw[k] * dvd[k];
    }
    if (nl) mu_pp[c] = mu_pp[c] + dtau * (dmdt + rmu_t[c]);
    if (WRFPLUS_TL) mu_pp_d[c] = mu_pp_d[c] + dtau * (dmdt_d + rmu_t_d[c]);
    if (write_mudf) {
        if (nl) mudf[c] = dmdt + rmu_t[c];
        if (WRFPLUS_TL) mudf_d[c] = dmdt_d + rmu_t_d[c];
    }
    wreal wl[WV_LEV], wd[WV_LEV];
    wreal total_ww = 0, total_d = 0;
    wl[0] = total_ww - ww_ref[c];
    wd[0] = -ww_ref_d[c];
    for (int kk = 1; kk < nz; ++kk) {
        int k = kk - 1;
        total_ww = total_ww - (dnw[k] * (c1h[k] * dmdt + dv[k] + c1h[k] * rmu_t[c])) / mc;
        total_d = total_d - dnw[k] * (c1h[k] * dmdt_d + dvd[k] + c1h[k] * rmu_t_d[c]) / mc;
        wl[kk] = total_ww - ww_ref[(size_t)kk * st + c];
        wd[kk] = total_d - ww_ref_d[(size_t)kk * st + c];
    }
    wl[nz] = (wreal)0.0; wd[nz] = 0;
    for (int kk = 0; kk <= nz; ++kk) {
        if (nl) ww_pp[(size_t)kk * st + c] = wl[kk];
        if (WRFPLUS_TL) ww_pp_d[(size_t)kk * st + c] = wd[kk];
    }
    wreal wdtn = 0, wdtn_d = 0;
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        wreal wdtn_up = 0, wdtn_up_d = 0;
        if (k + 1 < nz) {
            wdtn_up = wl[k + 1] * (fnm[k + 1] * thp[h + st] + fnp[k + 1] * thp[h]);
            wdtn_up_d = wd[k + 1] * (fnm[k + 1] * thp[h + st] + fnp[k + 1] * thp[h])
                      + wl[k + 1] * (fnm[k + 1] * thp_d[h + st] + fnp[k + 1] * thp_d[h]);
        }
        size_t hjp = MI3(k, jp, i, ny, nx), hjm = MI3(k, jm, i, ny, nx);
        size_t hip = MI3(k, j, ip, ny, nx), him = MI3(k, j, im, ny, nx);
        size_t vn = MI3(k, j + 1, i, ny + 1, nx), vs = MI3(k, j, i, ny + 1, nx);
        size_t ue = MI3(k, j, i + 1, ny, nx + 1), uw = MI3(k, j, i, ny, nx + 1);
        wreal hy = (wreal)0.5 * rdy * (v_pp[vn] * (thp[hjp] + thp[h])
                                       - v_pp[vs] * (thp[h] + thp[hjm]));
        wreal hx = (wreal)0.5 * rdx * (u_pp[ue] * (thp[hip] + thp[h])
                                       - u_pp[uw] * (thp[h] + thp[him]));
        wreal hy_d = (wreal)0.5 * rdy * (v_pp_d[vn] * (thp[hjp] + thp[h])
                                         + v_pp[vn] * (thp_d[hjp] + thp_d[h])
                                         - v_pp_d[vs] * (thp[h] + thp[hjm])
                                         - v_pp[vs] * (thp_d[h] + thp_d[hjm]));
        wreal hx_d = (wreal)0.5 * rdx * (u_pp_d[ue] * (thp[hip] + thp[h])
                                         + u_pp[ue] * (thp_d[hip] + thp_d[h])
                                         - u_pp_d[uw] * (thp[h] + thp[him])
                                         - u_pp[uw] * (thp_d[h] + thp_d[him]));
        if (nl) {
            wreal t = th_pp[h] + mc * dtau * rth_t[h];
            th_pp[h] = t - dtau * mc * (mc * (hy + hx) + rdnw[k] * (wdtn_up - wdtn));
        }
        if (WRFPLUS_TL) th_pp_d[h] = th_pp_d[h] + mc * dtau * rth_t_d[h]
                   - dtau * mc * (mc * (hy_d + hx_d) + rdnw[k] * (wdtn_up_d - wdtn_d));
        wdtn = wdtn_up; wdtn_d = wdtn_up_d;
    }
}

// Column adjoint: consumes the adjoints of the written entries, adds into
// rmu_t_a, rth_t_a, ww_ref_a, p_pp_a and the vertical part of thp_a, and
// leaves D and H (zero on the frame) for advance_mu_t_ad_gather.
extern "C" __global__
void advance_mu_t_ad(MT_STATE(_a), MT_TRAJ(), MT_TRAJ_ADJ, MT_PASSIVE,
                     wreal* __restrict__ D, wreal* __restrict__ H, MT_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx || nz + 1 > WV_LEV) return;
    int j = c / nx, i = c - j * nx;
    size_t st = (size_t)ny * nx;
    if (mt_frame(j, i, ny, nx, spec_zone)) {
        if (write_mudf) mudf_a[c] = 0;
        mu_pp_a[c] += mu_old_a[c]; mu_old_a[c] = 0;
        for (int k = 0; k < nz; ++k) {
            size_t h = (size_t)k * st + c;
            D[h] = 0; H[h] = 0;
            th_pp_a[h] += th_old_a[h]; th_old_a[h] = 0;
            p_pp_a[h] += p_old_a[h]; p_old_a[h] = 0;
        }
        return;
    }
    int ip = open_x ? min(i + 1, nx - 1) : (i + 1) % nx;
    int im = open_x ? max(i - 1, 0) : (i - 1 + nx) % nx;
    int jp = open_y ? min(j + 1, ny - 1) : (j + 1) % ny;
    int jm = open_y ? max(j - 1, 0) : (j - 1 + ny) % ny;
    size_t cw = (size_t)j * nx + im, ce = (size_t)j * nx + ip;
    size_t cs = (size_t)jm * nx + i, cn = (size_t)jp * nx + i;
    wreal muw = (wreal)0.5 * (mup[c] + mup[cw] + mub2d[c] + mub2d[cw]);
    wreal mue = (wreal)0.5 * (mup[ce] + mup[c] + mub2d[ce] + mub2d[c]);
    wreal mus = (wreal)0.5 * (mup[c] + mup[cs] + mub2d[c] + mub2d[cs]);
    wreal mun = (wreal)0.5 * (mup[cn] + mup[c] + mub2d[cn] + mub2d[c]);
    wreal mc = mapped ? msft[c] : (wreal)1.0;
    wreal uxw = mapped ? msfu[(size_t)j * (nx + 1) + i] : (wreal)1.0;
    wreal uxe = mapped ? msfu[(size_t)j * (nx + 1) + i + 1] : (wreal)1.0;
    wreal vys = mapped ? msfv[(size_t)j * nx + i] : (wreal)1.0;
    wreal vyn = mapped ? msfv[(size_t)(j + 1) * nx + i] : (wreal)1.0;

    // Forward recompute: the layer divergences and Omega''.
    wreal dv[WV_LEV], wl[WV_LEV];
    wreal dmdt = 0;
    for (int k = 0; k < nz; ++k) {
        size_t uw = MI3(k, j, i, ny, nx + 1), ue = uw + 1;
        size_t vs = MI3(k, j, i, ny + 1, nx), vn = vs + nx;
        wreal vs_full = v_pp[vs] + (c1h[k] * mus + c2h[k]) * v[vs] * vys;
        wreal vn_full = v_pp[vn] + (c1h[k] * mun + c2h[k]) * v[vn] * vyn;
        wreal uw_full = u_pp[uw] + ((c1h[k] * muw + c2h[k]) * u[uw]) / uxw;
        wreal ue_full = u_pp[ue] + ((c1h[k] * mue + c2h[k]) * u[ue]) / uxe;
        dv[k] = mc * mc * (rdy * (vn_full - vs_full) + rdx * (ue_full - uw_full));
        dmdt = dmdt + dnw[k] * dv[k];
    }
    wreal total_ww = 0;
    wl[0] = total_ww - ww_ref[c];
    for (int kk = 1; kk < nz; ++kk) {
        int k = kk - 1;
        total_ww = total_ww - (dnw[k] * (c1h[k] * dmdt + dv[k] + c1h[k] * rmu_t[c])) / mc;
        wl[kk] = total_ww - ww_ref[(size_t)kk * st + c];
    }

    // Reverse.  theta'': th = th0 + mc*dtau*rth - dtau*mc*(mc*(hy+hx)
    // + rdnw*(wdtn[k+1] - wdtn[k])), wdtn[m] = ww[m]*(fnm θ[m] + fnp θ[m-1]).
    wreal aww[WV_LEV];
    for (int kk = 0; kk <= nz; ++kk) aww[kk] = 0;
    wreal a_prev = 0;                      // dtau*mc*rdnw[k-1]*A[k-1]
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        wreal A = th_pp_a[h];
        rth_t_a[h] += mc * dtau * A;
        H[h] = -dtau * mc * mc * A;
        wreal e = dtau * mc * rdnw[k] * A;
        if (k >= 1) {
            wreal aW = e - a_prev;         // adjoint of wdtn[k]
            aww[k] = aW * (fnm[k] * thp[h] + fnp[k] * thp[h - st]);
            thp_a[h] += aW * wl[k] * fnm[k];
            thp_a[h - st] += aW * wl[k] * fnp[k];
        }
        a_prev = e;
        th_pp_a[h] = A + th_old_a[h]; th_old_a[h] = 0;
        p_pp_a[h] += p_old_a[h]; p_old_a[h] = 0;
    }
    // Omega'': ww[kk] = total[kk] - ww_ref[kk] (kk < nz), ww[nz] = 0.
    wreal adv[WV_LEV];
    wreal admdt = 0, armu = 0;
    ww_ref_a[c] -= ww_pp_a[c];
    ww_pp_a[c] = 0;
    ww_pp_a[(size_t)nz * st + c] = 0;
    wreal S = 0;
    for (int k = nz - 2; k >= 0; --k) {
        size_t w1 = (size_t)(k + 1) * st + c;
        wreal Wt = ww_pp_a[w1] + aww[k + 1];
        ww_ref_a[w1] -= Wt;
        ww_pp_a[w1] = 0;
        S += Wt;
        wreal coef = -dnw[k] * S / mc;
        adv[k] = coef;
        admdt += coef * c1h[k];
        armu += coef * c1h[k];
    }
    adv[nz - 1] = 0;
    if (write_mudf) {
        admdt += mudf_a[c]; armu += mudf_a[c];
        mudf_a[c] = 0;
    }
    wreal Am = mu_pp_a[c];
    admdt += dtau * Am; armu += dtau * Am;
    mu_pp_a[c] = Am + mu_old_a[c]; mu_old_a[c] = 0;
    rmu_t_a[c] += armu;
    for (int k = 0; k < nz; ++k) {
        wreal a = adv[k] + dnw[k] * admdt;
        D[(size_t)k * st + c] = a * mc * mc;
    }
}

// Face gather: column (j, i) writes the u faces i (and nx on the last
// column) and the v faces j (and ny on the last row), and gathers thp_a
// and mup_a from every face whose stencil reaches it (clamped/wrapped
// neighbours included).  A face f between columns f-1 and f carries
// a_full = rdx*(D[f-1] - D[f]) and the theta flux A = 0.5*rdx*u_pp*(H[f-1]
// - H[f]); columns outside the grid contribute zero.
// Face q (0-3) of cell p in one direction: p's own (west/south) face,
// p+1, and the clamped/wrapped boundary faces whose stencil reaches p
// (open: face 0 on p = 0, face n on p = n-1; periodic: face 0 on p = n-1,
// face n on p = 0).  -1: none.  The face p writes is q = 0, plus q = 1 on
// the last cell.
static __device__ __forceinline__ int mt_face(int q, int p, int n, int open)
{
    if (q == 0) return p;
    if (q == 1) return p + 1;
    if (q == 2) return (open ? p == 0 : p == n - 1) ? 0 : -1;
    return (open ? p == n - 1 : p == 0) ? n : -1;
}

static __device__ __forceinline__ int mt_lcell(int f, int n, int open)
{
    return f >= 1 ? f - 1 : (open ? 0 : n - 1);
}

static __device__ __forceinline__ int mt_rcell(int f, int n, int open)
{
    return f <= n - 1 ? f : (open ? n - 1 : 0);
}

extern "C" __global__
void advance_mu_t_ad_gather(MT_TRAJ(), MT_TRAJ_ADJ, MT_PASSIVE,
                            const wreal* __restrict__ D,
                            const wreal* __restrict__ H, MT_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    int j = c / nx, i = c - j * nx;
    size_t st = (size_t)ny * nx;
    wreal mfx[4], mfy[4], sx[4], sy[4], amx[4], amy[4];
    for (int q = 0; q < 4; ++q) {
        int f = mt_face(q, i, nx, open_x);
        amx[q] = 0; mfx[q] = 0; sx[q] = 1;
        if (f >= 0) {
            size_t L = (size_t)j * nx + mt_lcell(f, nx, open_x);
            size_t R = (size_t)j * nx + mt_rcell(f, nx, open_x);
            mfx[q] = (wreal)0.5 * (mup[R] + mup[L] + mub2d[R] + mub2d[L]);
            sx[q] = mapped ? msfu[(size_t)j * (nx + 1) + f] : (wreal)1.0;
        }
        int g = mt_face(q, j, ny, open_y);
        amy[q] = 0; mfy[q] = 0; sy[q] = 1;
        if (g >= 0) {
            size_t L = (size_t)mt_lcell(g, ny, open_y) * nx + i;
            size_t R = (size_t)mt_rcell(g, ny, open_y) * nx + i;
            mfy[q] = (wreal)0.5 * (mup[R] + mup[L] + mub2d[R] + mub2d[L]);
            sy[q] = mapped ? msfv[(size_t)g * nx + i] : (wreal)1.0;
        }
    }
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        wreal ath = 0;
        for (int q = 0; q < 4; ++q) {
            int f = mt_face(q, i, nx, open_x);
            if (f < 0) continue;
            wreal dl = f >= 1 ? D[MI3(k, j, f - 1, ny, nx)] : (wreal)0;
            wreal dr = f <= nx - 1 ? D[MI3(k, j, f, ny, nx)] : (wreal)0;
            wreal hl = f >= 1 ? H[MI3(k, j, f - 1, ny, nx)] : (wreal)0;
            wreal hr = f <= nx - 1 ? H[MI3(k, j, f, ny, nx)] : (wreal)0;
            size_t fi = MI3(k, j, f, ny, nx + 1);
            wreal afull = rdx * (dl - dr);
            wreal ah = (wreal)0.5 * rdx * (hl - hr);
            ath += ah * u_pp[fi];
            amx[q] += afull * c1h[k] * u[fi] / sx[q];
            if (q == 0 || (q == 1 && i == nx - 1)) {
                wreal ts = thp[MI3(k, j, mt_lcell(f, nx, open_x), ny, nx)]
                         + thp[MI3(k, j, mt_rcell(f, nx, open_x), ny, nx)];
                u_pp_a[fi] += afull + ah * ts;
                u_a[fi] += afull * (c1h[k] * mfx[q] + c2h[k]) / sx[q];
            }
        }
        for (int q = 0; q < 4; ++q) {
            int g = mt_face(q, j, ny, open_y);
            if (g < 0) continue;
            wreal dl = g >= 1 ? D[MI3(k, g - 1, i, ny, nx)] : (wreal)0;
            wreal dr = g <= ny - 1 ? D[MI3(k, g, i, ny, nx)] : (wreal)0;
            wreal hl = g >= 1 ? H[MI3(k, g - 1, i, ny, nx)] : (wreal)0;
            wreal hr = g <= ny - 1 ? H[MI3(k, g, i, ny, nx)] : (wreal)0;
            size_t fi = MI3(k, g, i, ny + 1, nx);
            wreal afull = rdy * (dl - dr);
            wreal ah = (wreal)0.5 * rdy * (hl - hr);
            ath += ah * v_pp[fi];
            amy[q] += afull * c1h[k] * v[fi] * sy[q];
            if (q == 0 || (q == 1 && j == ny - 1)) {
                wreal ts = thp[MI3(k, mt_lcell(g, ny, open_y), i, ny, nx)]
                         + thp[MI3(k, mt_rcell(g, ny, open_y), i, ny, nx)];
                v_pp_a[fi] += afull + ah * ts;
                v_a[fi] += afull * (c1h[k] * mfy[q] + c2h[k]) * sy[q];
            }
        }
        thp_a[h] += ath;
    }
    wreal am = 0;
    for (int q = 0; q < 4; ++q) am += amx[q] + amy[q];
    mup_a[c] += (wreal)0.5 * am;
}

// solve_em's specified-frame update after advance_mu_t: muts = MUT + mu''
// on the solved columns; on the frame mu'', theta'' and muts take
// dtau*tendency (spec_bdyupdate).  periodic_x frees the x frame.
static __device__ __forceinline__ bool mt_spec_frame(int j, int i, int ny,
                                                     int nx, int spec_zone,
                                                     int open_x)
{
    return j < spec_zone || j >= ny - spec_zone
        || (open_x && (i < spec_zone || i >= nx - spec_zone));
}

extern "C" __global__
void frame_mu_t_tl(wreal* __restrict__ mu_pp, wreal* __restrict__ th_pp,
                   wreal* __restrict__ muts, wreal* __restrict__ mu_pp_d,
                   wreal* __restrict__ th_pp_d, wreal* __restrict__ muts_d,
                   const wreal* __restrict__ rmu_t, const wreal* __restrict__ rth_t,
                   const wreal* __restrict__ mup, const wreal* __restrict__ rmu_t_d,
                   const wreal* __restrict__ rth_t_d, const wreal* __restrict__ mup_d,
                   const wreal* __restrict__ mub2d, int nl, MT_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    int j = c / nx, i = c - j * nx;
    if (!mt_spec_frame(j, i, ny, nx, spec_zone, open_x)) {
        if (nl) muts[c] = (mub2d[c] + mup[c]) + mu_pp[c];
        if (WRFPLUS_TL) muts_d[c] = mup_d[c] + mu_pp_d[c];
        return;
    }
    if (nl) {
        muts[c] = muts[c] + dtau * rmu_t[c];
        mu_pp[c] = mu_pp[c] + dtau * rmu_t[c];
    }
    if (WRFPLUS_TL) muts_d[c] = muts_d[c] + dtau * rmu_t_d[c];
    if (WRFPLUS_TL) mu_pp_d[c] = mu_pp_d[c] + dtau * rmu_t_d[c];
    size_t st = (size_t)ny * nx;
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        if (nl) th_pp[h] = th_pp[h] + dtau * rth_t[h];
        if (WRFPLUS_TL) th_pp_d[h] = th_pp_d[h] + dtau * rth_t_d[h];
    }
}

extern "C" __global__
void frame_mu_t_ad(wreal* __restrict__ mu_pp_a, wreal* __restrict__ th_pp_a,
                   wreal* __restrict__ muts_a, wreal* __restrict__ rmu_t_a,
                   wreal* __restrict__ rth_t_a, wreal* __restrict__ mup_a,
                   MT_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    int j = c / nx, i = c - j * nx;
    if (!mt_spec_frame(j, i, ny, nx, spec_zone, open_x)) {
        mu_pp_a[c] += muts_a[c];
        mup_a[c] += muts_a[c];
        muts_a[c] = 0;
        return;
    }
    rmu_t_a[c] += dtau * (muts_a[c] + mu_pp_a[c]);
    size_t st = (size_t)ny * nx;
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        rth_t_a[h] += dtau * th_pp_a[h];
    }
}
