// woof/core/kernels/nest_top.cu
//
// Partial-depth nest, Phase D2 (plans/sorted-mapping-hummingbird.md): the
// child's dynamics with a specified, nonzero mass flux through its top.
//
// The child is the parent's own discrete column cut at a pure-pressure full
// level k_top (woof/core/partial_nest.py); its nz is k_top and its full
// level nz is the interface.  These are NEW entry points only.  The loader
// assembles this unit AFTER acoustic.cu (kernels/__init__.py
// _EXTRA_HEADERS["nest_top"]), so the device helpers below -- pdmp,
// dpn_face, pgrad_face, cq_pair, diagnose_p_column -- are acoustic.cu's own
// text and every existing kernel and its launch stays byte-identical.
// Each entry point is the default-arithmetic branch of the acoustic.cu (or
// dycore.py) routine it is named after, statement for statement, with only
// the top changed; the strict GPUWM_WRF_EXACT* builds are refused by the
// Python driver (woof/core/partial_nest_dyn.py) before this unit compiles.
//
// Sign convention (WRF calc_ww_cp / advance_mu_t).  ww is the coupled eta
// mass flux Omega, Omega-signed: positive is downward (eta decreases with
// height).  The recurrence ww(k) = ww(k-1) - dnw(k-1)*(c1h(k-1)*dmdt +
// div(k-1)) with dmdt = sum(dnw*div) closes ww(nz) = 0 on a full column,
// because sum(dnw*c1h) = B(top) - B(sfc) = -1.  On the child the same sum
// is still -1 (B = 0 at the pure-pressure interface), so a top flux ww_top
// requires dmdt = sum_{k<k_top}(dnw*div) + ww_top: a positive (downward)
// flux through the top adds mass to the child's column.  That is the only
// change to the mass equation.
//
// "Specified at the top" arguments, all per child column (ny, nx):
//   ww_top_pp  the perturbation (acoustic) eta flux at the interface;
//   ww_top_rf  the stage-reference eta flux there (the big-step ww);
//   dmdt_adj_* an additive column-mass-tendency adjustment, zero for a
//              free nest.  It carries anything that enters the column sum
//              but not the face flux: the self-consistency oracle uses it
//              to absorb the reassociation of the column sum (a full
//              column adds its levels above k_top one by one, the child
//              adds their total once), and it is D3's hook for a mass
//              correction.  rmu_t keeps the existing kernels' convention:
//              it enters dmdt + rmu_t exactly as on a full column.
//   p_above    p'' (and its history p_above_old) at the parent half level
//              k_top, for the pressure-gradient dpn at the interface;
//   t_above    theta_t* = thb + thp at the parent half level k_top;
//   rhs_top    the explicit phi'' predictor at the interface (computed by
//              nest_top_ph_predictor from the parent's ww and phi_t* one
//              level above, or zero for a rigid lid);
//   w_top, ph_top  the solved w'' and phi'' at the interface.
// Interface coordinate scalars fnm_top/fnp_top/rdnw_top are the parent's
// fnm[k_top], fnp[k_top], rdnw[k_top].

// dpn at full level kf, with the interface (kf == nz) taking the parent's
// interior form from the specified p'' above.
static __device__
real nest_dpn_face(const real* __restrict__ p, const real* __restrict__ pold,
                   real smdiv, int kf, int nz, size_t cA, size_t cB, size_t st,
                   const real* __restrict__ fnm, const real* __restrict__ fnp,
                   real cf1, real cf2, real cf3, int top_lid,
                   const real* __restrict__ p_above,
                   const real* __restrict__ p_above_old,
                   real fnm_top, real fnp_top)
{
    if (kf == nz)
        return 0.5f * (fnm_top * (pdmp(p_above, p_above_old, cA, smdiv)
                                + pdmp(p_above, p_above_old, cB, smdiv))
                     + fnp_top * (pdmp(p, pold, cA + (size_t)(kf-1) * st, smdiv)
                                + pdmp(p, pold, cB + (size_t)(kf-1) * st, smdiv)));
    return dpn_face(p, pold, smdiv, kf, nz, cA, cB, st, fnm, fnp,
                    cf1, cf2, cf3, top_lid);
}

// pgrad_face (default build) with the interface dpn above.
static __device__
real nest_pgrad_face(size_t cA, size_t cB, int k, int nz, size_t st, real rd,
                     const real* __restrict__ ph_pp, const real* __restrict__ php,
                     const real* __restrict__ phb,
                     const real* __restrict__ alt,
                     const real* __restrict__ al_pp, const real* __restrict__ pb,
                     const real* __restrict__ p_pp, const real* __restrict__ p_old,
                     real smdiv,
                     const real* __restrict__ mup, const real* __restrict__ mu_pp,
                     const real* __restrict__ mub2d,
                     const real* __restrict__ c1h, const real* __restrict__ c2h,
                     const real* __restrict__ fnm, const real* __restrict__ fnp,
                     const real* __restrict__ rdnw,
                     real cf1, real cf2, real cf3, int base3d, int top_lid,
                     const real* __restrict__ p_above,
                     const real* __restrict__ p_above_old,
                     real fnm_top, real fnp_top)
{
    real muf = 0.5f * ((mub2d[cB] + mup[cB]) + (mub2d[cA] + mup[cA]));
    size_t a = cA + (size_t)k * st, b = cB + (size_t)k * st;
    real dph = (ph_pp[a + st] - ph_pp[b + st]) + (ph_pp[a] - ph_pp[b]);
    real dpe = pdmp(p_pp, p_old, a, smdiv) - pdmp(p_pp, p_old, b, smdiv);
    real dpb = base3d ? (pb[a] - pb[b]) : 0.0f;
    real dpxy = 0.5f * rd * (c1h[k] * muf + c2h[k])
              * (dph + (alt[a] + alt[b]) * dpe + (al_pp[a] + al_pp[b]) * dpb);
    real dpn_up = nest_dpn_face(p_pp, p_old, smdiv, k + 1, nz, cA, cB, st,
                                fnm, fnp, cf1, cf2, cf3, top_lid,
                                p_above, p_above_old, fnm_top, fnp_top);
    real dpn_dn = nest_dpn_face(p_pp, p_old, smdiv, k,     nz, cA, cB, st,
                                fnm, fnp, cf1, cf2, cf3, top_lid,
                                p_above, p_above_old, fnm_top, fnp_top);
    real dphp = 0.5f * ((php[a] + php[a + st]) - (php[b] + php[b + st]));
    if (base3d)
        dphp += 0.5f * ((phb[a] + phb[a + st]) - (phb[b] + phb[b + st]));
    dpxy += rd * dphp * (rdnw[k] * (dpn_up - dpn_dn)
                         - 0.5f * c1h[k] * (mu_pp[cB] + mu_pp[cA]));
    return dpxy;
}

// advance_uv (default build) on the child: the top half level's pressure
// gradient sees the parent's p'' above the interface.
extern "C" __global__
void advance_uv_top(real* __restrict__ u_pp, real* __restrict__ v_pp,
                    const real* __restrict__ ru_t, const real* __restrict__ rv_t,
                    const real* __restrict__ p_pp,
                    const real* __restrict__ p_pp_old,
                    const real* __restrict__ ph_pp, const real* __restrict__ php,
                    const real* __restrict__ phb,
                    const real* __restrict__ alt,
                    const real* __restrict__ al_pp, const real* __restrict__ pb,
                    const real* __restrict__ mup, const real* __restrict__ mu_pp,
                    const real* __restrict__ mub2d,
                    const real* __restrict__ c1h, const real* __restrict__ c2h,
                    const real* __restrict__ fnm, const real* __restrict__ fnp,
                    const real* __restrict__ rdnw,
                    const real* __restrict__ cqu,
                    const real* __restrict__ cqv, int moist_cq,
                    real cf1, real cf2, real cf3,
                    int top_lid,
                    real rdx, real rdy, real dtau, real smdiv,
                    int spec_zone, int base3d, int nz, int ny, int nx,
                    const real* __restrict__ p_above,
                    const real* __restrict__ p_above_old,
                    real fnm_top, real fnp_top)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;
    size_t st = (size_t)ny * nx;

    if (j < ny
        && j >= spec_zone && j < ny - spec_zone
        && i >= spec_zone && i <= nx - spec_zone) {
        int iA = (i == nx ? 0 : i), iB = (i == 0 ? nx - 1 : i - 1);
        size_t cA = (size_t)j * nx + iA, cB = (size_t)j * nx + iB;
        real dpxy = nest_pgrad_face(cA, cB, k, nz, st, rdx, ph_pp, php, phb, alt,
                                    al_pp, pb, p_pp, p_pp_old, smdiv,
                                    mup, mu_pp, mub2d, c1h, c2h,
                                    fnm, fnp, rdnw, cf1, cf2, cf3, base3d,
                                    top_lid, p_above, p_above_old,
                                    fnm_top, fnp_top);
        size_t uix = I3S(k, j, i, ny, nxf);
        if (moist_cq)
            u_pp[uix] += dtau * (ru_t[uix] - cqu[uix] * dpxy);
        else
            u_pp[uix] += dtau * (ru_t[uix] - dpxy);
    } else if (j < ny) {
        size_t uix = I3S(k, j, i, ny, nxf);
        u_pp[uix] = arn_add(u_pp[uix], arn_mul(dtau, ru_t[uix]));
    }
    if (i < nx
        && i >= spec_zone && i < nx - spec_zone
        && j >= spec_zone && j <= ny - spec_zone) {
        int jA = (j == ny ? 0 : j), jB = (j == 0 ? ny - 1 : j - 1);
        size_t cA = (size_t)jA * nx + i, cB = (size_t)jB * nx + i;
        real dpxy = nest_pgrad_face(cA, cB, k, nz, st, rdy, ph_pp, php, phb, alt,
                                    al_pp, pb, p_pp, p_pp_old, smdiv,
                                    mup, mu_pp, mub2d, c1h, c2h,
                                    fnm, fnp, rdnw, cf1, cf2, cf3, base3d,
                                    top_lid, p_above, p_above_old,
                                    fnm_top, fnp_top);
        size_t vix = I3S(k, j, i, nyf, nx);
        if (moist_cq)
            v_pp[vix] += dtau * (rv_t[vix] - cqv[vix] * dpxy);
        else
            v_pp[vix] += dtau * (rv_t[vix] - dpxy);
    } else if (i < nx) {
        size_t vix = I3S(k, j, i, nyf, nx);
        v_pp[vix] = arn_add(v_pp[vix], arn_mul(dtau, rv_t[vix]));
    }
}

// advance_mu_th (default build) on the child: the column mass tendency
// gains the specified top flux (dmdt = sum below + ww_top), the Omega''
// recurrence integrates up to the interface and the interface face holds
// the specified ww_top_pp, and the top half level's vertical theta flux
// passes through it.  top_theta_upwind = 0: the parent's own centered face
// value fnm*theta_above + fnp*theta_below (what a full column computes);
// 1: upwinded on the total interface flux ww_top_rf + ww_top_pp (positive,
// downward: the parent's theta above enters).
extern "C" __global__
void advance_mu_th_top(const real* __restrict__ u_pp,
                       const real* __restrict__ v_pp,
                       const real* __restrict__ u, const real* __restrict__ v,
                       const real* __restrict__ mup,
                       real* __restrict__ mu_pp,
                       real* __restrict__ mu_pp_old,
                       const real* __restrict__ rmu_t,
                       real* __restrict__ mudf, int write_mudf,
                       const real* __restrict__ thp,
                       const real* __restrict__ thb,
                       real* __restrict__ th_pp,
                       real* __restrict__ th_pp_old,
                       const real* __restrict__ rth_t,
                       real* __restrict__ ww_pp,
                       const real* __restrict__ p_pp,
                       real* __restrict__ p_pp_old,
                       const real* __restrict__ dnw,
                       const real* __restrict__ rdnw,
                       const real* __restrict__ fnm,
                       const real* __restrict__ fnp,
                       const real* __restrict__ c1h, const real* __restrict__ c2h,
                       const real* __restrict__ mub2d,
                       real rdx, real rdy, real dtau,
                       int base3d, int nz, int ny, int nx,
                       int open_x, int open_y, int spec_zone,
                       const real* __restrict__ ww_top_pp,
                       const real* __restrict__ ww_top_rf,
                       const real* __restrict__ dmdt_adj_pp,
                       const real* __restrict__ dmdt_adj_rf,
                       const real* __restrict__ t_above,
                       real fnm_top, real fnp_top, int top_theta_upwind)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    int j = c / nx, i = c - j * nx;
    size_t st = (size_t)ny * nx;
    mu_pp_old[c] = mu_pp[c];
    for (int k = 0; k < nz; ++k) {
        size_t ci = (size_t)k * st + c;
        th_pp_old[ci] = th_pp[ci];
        p_pp_old[ci] = p_pp[ci];
    }
    if (i < spec_zone || i >= nx - spec_zone
        || j < spec_zone || j >= ny - spec_zone) {
        real rmu = rmu_t[c];
        if (write_mudf) mudf[c] = 0.0f;
        mu_pp[c] = arn_add(mu_pp[c], arn_mul(dtau, rmu));
        for (int k = 0; k < nz; ++k) {
            size_t ci = (size_t)k * st + c;
            real mt0 = arn_mul(300.0f, c1h[k]);
            real mt1 = arn_mul(mt0, rmu);
            real full_tendency = arn_add(rth_t[ci], mt1);
            th_pp[ci] = arn_add(th_pp[ci], arn_mul(dtau, full_tendency));
        }
        return;
    }
    int ip = open_x ? min(i + 1, nx - 1) : (i + 1) % nx;
    int im = open_x ? max(i - 1, 0)      : (i - 1 + nx) % nx;
    int jp = open_y ? min(j + 1, ny - 1) : (j + 1) % ny;
    int jm = open_y ? max(j - 1, 0)      : (j - 1 + ny) % ny;
    int nxf = nx + 1;
    size_t bstr = base3d ? st : 1;

    size_t cw = (size_t)j * nx + im, ce = (size_t)j * nx + ip;
    size_t cs = (size_t)jm * nx + i, cn = (size_t)jp * nx + i;
    real muw = 0.5f * ((mub2d[cw] + mup[cw]) + (mub2d[c] + mup[c]));
    real mue = 0.5f * ((mub2d[c] + mup[c]) + (mub2d[ce] + mup[ce]));
    real mus = 0.5f * ((mub2d[cs] + mup[cs]) + (mub2d[c] + mup[c]));
    real mun = 0.5f * ((mub2d[c] + mup[c]) + (mub2d[cn] + mup[cn]));

    real dmdt_pp = 0.0f, dmdt_rf = 0.0f;
    for (int k = 0; k < nz; ++k) {
        size_t uw = I3S(k, j, i, ny, nxf), ue = I3S(k, j, i + 1, ny, nxf);
        size_t vs = I3S(k, j, i, ny + 1, nx), vn = I3S(k, j + 1, i, ny + 1, nx);
        real dvp = rdx * (u_pp[ue] - u_pp[uw]) + rdy * (v_pp[vn] - v_pp[vs]);
        real dvr = rdx * ((c1h[k] * mue + c2h[k]) * u[ue]
                        - (c1h[k] * muw + c2h[k]) * u[uw])
                 + rdy * ((c1h[k] * mun + c2h[k]) * v[vn]
                        - (c1h[k] * mus + c2h[k]) * v[vs]);
        dmdt_pp += dnw[k] * dvp;
        dmdt_rf += dnw[k] * dvr;
    }
    // The flux through the top (module comment): each part of the column
    // sum gains its own interface flux, then the adjustment.
    real wtop = ww_top_pp[c];
    dmdt_pp = arn_add(arn_add(dmdt_pp, wtop), dmdt_adj_pp[c]);
    dmdt_rf = arn_add(arn_add(dmdt_rf, ww_top_rf[c]), dmdt_adj_rf[c]);
    real rmu = rmu_t[c];
    real mass_tendency = arn_add(arn_add(dmdt_pp, dmdt_rf), rmu);
    if (write_mudf) mudf[c] = mass_tendency;
    mu_pp[c] = arn_add(mu_pp[c], arn_mul(dtau, mass_tendency));

    ww_pp[c] = 0.0f;
    real wwk = 0.0f;
    for (int kk = 1; kk < nz; ++kk) {
        int k = kk - 1;
        size_t uw = I3S(k, j, i, ny, nxf), ue = I3S(k, j, i + 1, ny, nxf);
        size_t vs = I3S(k, j, i, ny + 1, nx), vn = I3S(k, j + 1, i, ny + 1, nx);
        real dvp = rdx * (u_pp[ue] - u_pp[uw]) + rdy * (v_pp[vn] - v_pp[vs]);
        wwk -= dnw[k] * (c1h[k] * (dmdt_pp + rmu) + dvp);
        ww_pp[(size_t)kk * st + c] = wwk;
    }
    ww_pp[(size_t)nz * st + c] = wtop;

    real wdtn_k = 0.0f;
    for (int k = 0; k < nz; ++k) {
        real wdtn_kp1;
        real t_dn = thb[(size_t)k * bstr + (base3d ? (size_t)c : 0)]
                  + thp[(size_t)k * st + c];
        if (k + 1 < nz) {
            real t_up = thb[(size_t)(k + 1) * bstr + (base3d ? (size_t)c : 0)]
                      + thp[(size_t)(k + 1) * st + c];
            wdtn_kp1 = ww_pp[(size_t)(k + 1) * st + c]
                     * (fnm[k + 1] * t_up + fnp[k + 1] * t_dn);
        } else if (top_theta_upwind) {
            real total = arn_add(ww_top_rf[c], wtop);
            wdtn_kp1 = wtop * (total > 0.0f ? t_above[c] : t_dn);
        } else {
            wdtn_kp1 = wtop * (fnm_top * t_above[c] + fnp_top * t_dn);
        }
        size_t bk = (size_t)k * bstr;
        real t_c  = thb[bk + (base3d ? (size_t)c : 0)]
                  + thp[(size_t)k * st + c];
        real t_ip = thb[bk + (base3d ? ce : 0)] + thp[IDX3(k, j, ip)];
        real t_im = thb[bk + (base3d ? cw : 0)] + thp[IDX3(k, j, im)];
        real t_jp = thb[bk + (base3d ? cn : 0)] + thp[IDX3(k, jp, i)];
        real t_jm = thb[bk + (base3d ? cs : 0)] + thp[IDX3(k, jm, i)];
        real hx = 0.5f * rdx * (u_pp[I3S(k, j, i + 1, ny, nxf)] * (t_ip + t_c)
                              - u_pp[I3S(k, j, i,     ny, nxf)] * (t_c + t_im));
        real hy = 0.5f * rdy * (v_pp[I3S(k, j + 1, i, ny + 1, nx)] * (t_jp + t_c)
                              - v_pp[I3S(k, j,     i, ny + 1, nx)] * (t_c + t_jm));
        size_t ci = (size_t)k * st + c;
        th_pp[ci] += dtau * (rth_t[ci]
                             - (hx + hy + rdnw[k] * (wdtn_kp1 - wdtn_k)));
        wdtn_k = wdtn_kp1;
    }
}

// The explicit phi'' predictor at the interface: advance_w_phi's rhs[nz] of
// a column that continues above, from the child's own levels nz-1, nz and
// the parent's eta flux ww_above and t* geopotential pht_above = phb + php
// at full level k_top + 1.  Run after advance_mu_th_top (ww_pp current) and
// before advance_w_phi_top (w_pp, ph_pp still the previous substep's).
extern "C" __global__
void nest_top_ph_predictor(real* __restrict__ rhs_top,
                           const real* __restrict__ w_pp,
                           const real* __restrict__ ph_pp,
                           const real* __restrict__ rph_t,
                           const real* __restrict__ ww_pp,
                           const real* __restrict__ php,
                           const real* __restrict__ phb,
                           const real* __restrict__ mup,
                           const real* __restrict__ mub2d,
                           const real* __restrict__ rdnw,
                           const real* __restrict__ c1f,
                           const real* __restrict__ c2f,
                           const real* __restrict__ ww_above,
                           const real* __restrict__ pht_above,
                           real fnm_top, real fnp_top, real rdnw_top,
                           real dtau, real epssm, int base3d,
                           int nz, int ny, int nx)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    size_t st = (size_t)ny * nx;
    size_t bstr = base3d ? st : 1;
    size_t boff = base3d ? (size_t)c : 0;
    real mut = mub2d[c] + mup[c];
    size_t f = (size_t)nz * st + c;
    real rhs = dtau * (rph_t[f] + 0.5f * G * (1.0f - epssm) * w_pp[f]);
    real ph_lo = phb[(size_t)(nz - 1) * bstr + boff]
               + php[(size_t)(nz - 1) * st + c];
    real ph_hi = phb[(size_t)nz * bstr + boff] + php[f];
    real wd_lo = 0.5f * (ww_pp[f] + ww_pp[f - st])
               * rdnw[nz - 1] * (ph_hi - ph_lo);
    real wd_hi = 0.5f * (ww_above[c] + ww_pp[f])
               * rdnw_top * (pht_above[c] - ph_hi);
    rhs -= dtau * (fnm_top * wd_hi + fnp_top * wd_lo);
    rhs_top[c] = ph_pp[f] + rhs / (c1f[nz] * mut + c2f[nz]);
}

// advance_w_phi (default build) on the child with w'' and phi'' specified
// at the interface: the predictor there is rhs_top, the Thomas
// back-substitution starts from w_top, and phi''(nz) = ph_top.  The
// coefficients come from the existing calc_coefs run on the child (rows
// below the interface are a full column's; the child's row nz is unused).
// No damp_opt=3 damper (its layer is measured from the PARENT's top; the
// driver refuses it).
extern "C" __global__
void advance_w_phi_top(real* __restrict__ w_pp, real* __restrict__ ph_pp,
                       const real* __restrict__ rw_t,
                       const real* __restrict__ rph_t,
                       const real* __restrict__ ww_pp,
                       const real* __restrict__ mu_pp,
                       const real* __restrict__ mu_pp_old,
                       const real* __restrict__ th_pp,
                       const real* __restrict__ th_pp_old,
                       const real* __restrict__ thp, const real* __restrict__ thb,
                       const real* __restrict__ php, const real* __restrict__ phb,
                       const real* __restrict__ alt, const real* __restrict__ c2a,
                       const real* __restrict__ a, const real* __restrict__ alpha,
                       const real* __restrict__ gam,
                       const real* __restrict__ mup,
                       const real* __restrict__ u_pp,
                       const real* __restrict__ v_pp,
                       const real* __restrict__ ht,
                       const real* __restrict__ rdn, const real* __restrict__ rdnw,
                       const real* __restrict__ fnm, const real* __restrict__ fnp,
                       const real* __restrict__ c1h, const real* __restrict__ c2h,
                       const real* __restrict__ c1f, const real* __restrict__ c2f,
                       const real* __restrict__ mub2d,
                       const real* __restrict__ cqw, int moist_cq,
                       real* __restrict__ p_pp, real* __restrict__ al_pp,
                       real cf1, real cf2, real cf3, real rdx, real rdy,
                       real dtau, real epssm,
                       int boundary_x, int boundary_y,
                       int spec_zone, int base3d,
                       int nz, int ny, int nx,
                       const real* __restrict__ rhs_top,
                       const real* __restrict__ w_top,
                       const real* __restrict__ ph_top)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx || nz + 1 > WPHI_MAX_LEV) return;
    int j = c / nx, i = c - j * nx;
    if (i < spec_zone || i >= nx - spec_zone
        || j < spec_zone || j >= ny - spec_zone) return;
    size_t st = (size_t)ny * nx;
    size_t bstr = base3d ? st : 1;
    size_t boff = base3d ? (size_t)c : 0;
    real mut = mub2d[c] + mup[c];
    real muts = mut + mu_pp[c];
    real muave = 0.5f * ((1.0f + epssm) * mu_pp[c]
                       + (1.0f - epssm) * mu_pp_old[c]);

    real rhs[WPHI_MAX_LEV];
    rhs[0] = 0.0f;
    for (int k = 1; k < nz; ++k) {
        size_t f = (size_t)k * st + c;
        rhs[k] = dtau * (rph_t[f]
                         + 0.5f * G * (1.0f - epssm) * w_pp[f]);
    }
    real ph_lo = phb[boff] + php[c];
    real ph_hi = phb[bstr + boff] + php[(size_t)st + c];
    real wd_lo = 0.5f * (ww_pp[(size_t)st + c] + ww_pp[c])
               * rdnw[0] * (ph_hi - ph_lo);
    for (int k = 1; k < nz; ++k) {
        ph_lo = ph_hi;
        ph_hi = phb[(size_t)(k + 1) * bstr + boff]
              + php[(size_t)(k + 1) * st + c];
        real wd_hi = 0.5f * (ww_pp[(size_t)(k + 1) * st + c]
                           + ww_pp[(size_t)k * st + c])
                   * rdnw[k] * (ph_hi - ph_lo);
        rhs[k] -= dtau * (fnm[k] * wd_hi + fnp[k] * wd_lo);
        wd_lo = wd_hi;
    }
    for (int k = 1; k < nz; ++k)
        rhs[k] = ph_pp[(size_t)k * st + c]
               + rhs[k] / (c1f[k] * mut + c2f[k]);
    rhs[nz] = rhs_top[c];

    {
        int ip = boundary_x ? min(i + 1, nx - 1) : (i + 1) % nx;
        int im = boundary_x ? max(i - 1, 0)      : (i - 1 + nx) % nx;
        int jp = boundary_y ? min(j + 1, ny - 1) : (j + 1) % ny;
        int jm = boundary_y ? max(j - 1, 0)      : (j - 1 + ny) % ny;
        int nxf = nx + 1;
        size_t cip = (size_t)j * nx + ip, cim = (size_t)j * nx + im;
        size_t cjp = (size_t)jp * nx + i, cjm = (size_t)jm * nx + i;
        real ue = cf1 * u_pp[I3S(0, j, i + 1, ny, nxf)]
                + cf2 * u_pp[I3S(1, j, i + 1, ny, nxf)]
                + cf3 * u_pp[I3S(2, j, i + 1, ny, nxf)];
        real uw = cf1 * u_pp[I3S(0, j, i, ny, nxf)]
                + cf2 * u_pp[I3S(1, j, i, ny, nxf)]
                + cf3 * u_pp[I3S(2, j, i, ny, nxf)];
        real vn = cf1 * v_pp[I3S(0, j + 1, i, ny + 1, nx)]
                + cf2 * v_pp[I3S(1, j + 1, i, ny + 1, nx)]
                + cf3 * v_pp[I3S(2, j + 1, i, ny + 1, nx)];
        real vs = cf1 * v_pp[I3S(0, j, i, ny + 1, nx)]
                + cf2 * v_pp[I3S(1, j, i, ny + 1, nx)]
                + cf3 * v_pp[I3S(2, j, i, ny + 1, nx)];
        w_pp[c] = 0.5f * rdy * ((ht[cjp] - ht[c]) * vn
                                + (ht[c] - ht[cjm]) * vs)
                + 0.5f * rdx * ((ht[cip] - ht[c]) * ue
                                + (ht[c] - ht[cim]) * uw);
    }

    real t2_dn = 0.5f * ((1.0f + epssm) * th_pp[c]
                       + (1.0f - epssm) * th_pp_old[c])
               / ((c1h[0] * muts + c2h[0]) * (thb[boff] + thp[c]));
    for (int k = 1; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        size_t hm = h - st;
        real t2_up = 0.5f * ((1.0f + epssm) * th_pp[h]
                           + (1.0f - epssm) * th_pp_old[h])
                   / ((c1h[k] * muts + c2h[k])
                      * (thb[(size_t)k * bstr + boff] + thp[h]));
        real dph_up = (1.0f + epssm) * (rhs[k + 1] - rhs[k])
                    + (1.0f - epssm) * (ph_pp[(size_t)(k + 1) * st + c]
                                        - ph_pp[(size_t)k * st + c]);
        real dph_dn = (1.0f + epssm) * (rhs[k] - rhs[k - 1])
                    + (1.0f - epssm) * (ph_pp[(size_t)k * st + c]
                                        - ph_pp[(size_t)(k - 1) * st + c]);
        size_t f = (size_t)k * st + c;
        if (moist_cq) {
            w_pp[f] += dtau * rw_t[f]
                     + cqw[f] * (0.5f * dtau * G * rdn[k]
                       * (c2a[h] * rdnw[k]
                          / (c1h[k] * mut + c2h[k]) * dph_up
                          - c2a[hm] * rdnw[k - 1]
                            / (c1h[k - 1] * mut + c2h[k - 1]) * dph_dn))
                     + dtau * G * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                             - c2a[hm] * alt[hm] * t2_dn)
                                   - c1f[k] * muave);
        } else {
            w_pp[f] += dtau * rw_t[f]
                     + 0.5f * dtau * G * rdn[k]
                       * (c2a[h] * rdnw[k]
                          / (c1h[k] * mut + c2h[k]) * dph_up
                          - c2a[hm] * rdnw[k - 1]
                            / (c1h[k - 1] * mut + c2h[k - 1]) * dph_dn)
                     + dtau * G * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                             - c2a[hm] * alt[hm] * t2_dn)
                                   - c1f[k] * muave);
        }
        t2_dn = t2_up;
    }

    // Thomas solve: forward elimination below the interface, then the
    // back-substitution from the specified w''(nz).
    for (int k = 1; k < nz; ++k)
        w_pp[(size_t)k * st + c] =
            (w_pp[(size_t)k * st + c]
             - a[(size_t)k * st + c] * w_pp[(size_t)(k - 1) * st + c])
            * alpha[(size_t)k * st + c];
    w_pp[(size_t)nz * st + c] = w_top[c];
    for (int k = nz - 1; k >= 1; --k)
        w_pp[(size_t)k * st + c] -= gam[(size_t)k * st + c]
                                  * w_pp[(size_t)(k + 1) * st + c];

    for (int k = 1; k < nz; ++k)
        ph_pp[(size_t)k * st + c] =
            rhs[k] + 0.5f * dtau * G * (1.0f + epssm)
                     * w_pp[(size_t)k * st + c]
                     / (c1f[k] * muts + c2f[k]);
    ph_pp[(size_t)nz * st + c] = ph_top[c];

    diagnose_p_column(p_pp, al_pp, th_pp, ph_pp, mu_pp, thp, thb, alt,
                      c2a, mup, rdnw, c1h, c2h, mub2d,
                      (size_t)c, st, base3d, nz);
}

// Big-step Omega (dycore._omega_column_kernel, WRF calc_ww_cp) on the
// child, with the specified interface flux: dmdt = (sum below + ww_top) +
// dmdt_adj, then the same recurrence, and ww(nz) = ww_top.  Every product
// and sum boundary is an explicit round-to-nearest intrinsic, as there.
extern "C" __global__
void calc_ww_top(const real* __restrict__ ru, const real* __restrict__ rv,
                 const real* __restrict__ dnw, const real* __restrict__ c1h,
                 const real* __restrict__ msft, int has_msf,
                 real rdx, real rdy, int nz, int ny, int nx,
                 const real* __restrict__ ww_top,
                 const real* __restrict__ dmdt_adj,
                 real* __restrict__ ww)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int ncol = ny * nx;
    if (col >= ncol) return;
    const int jj = col / nx;
    const int ii = col - jj * nx;
    real dmdt = 0.0f;
    for (int k = 0; k < nz; ++k) {
        const int u0 = (k * ny + jj) * (nx + 1) + ii;
        const int v0 = (k * (ny + 1) + jj) * nx + ii;
        const real du = __fsub_rn(ru[u0 + 1], ru[u0]);
        const real dv = __fsub_rn(rv[v0 + nx], rv[v0]);
        const real bracket = __fadd_rn(__fmul_rn(rdx, du), __fmul_rn(rdy, dv));
        const real weight = has_msf ? __fmul_rn(msft[col], dnw[k]) : dnw[k];
        const real divv = __fmul_rn(weight, bracket);
        dmdt = __fadd_rn(dmdt, divv);
        ww[(k + 1) * ncol + col] = divv;
    }
    dmdt = __fadd_rn(__fadd_rn(dmdt, ww_top[col]), dmdt_adj[col]);
    real w = 0.0f;
    ww[col] = w;
    for (int k = 1; k < nz; ++k) {
        const int at = k * ncol + col;
        const real divv = ww[at];
        w = __fsub_rn(w, __fmul_rn(__fmul_rn(dnw[k - 1], c1h[k - 1]), dmdt));
        w = __fsub_rn(w, divv);
        ww[at] = w;
    }
    ww[nz * ncol + col] = ww_top[col];
}

// Column divergence sums the two dmdt recurrences above form, over the
// levels below klev and over the whole column (sums[0..3] = acoustic
// perturbation and reference parts below klev, then over all nz; sums[4..5]
// = calc_ww's below klev and over all nz), in exactly the arithmetic of
// advance_mu_th (default build) and dycore._omega_column_kernel.  Used to
// state, word for word, the column-mass reassociation a cut column makes
// (the self-consistency oracle, tests/test_partial_nest_dyn.py) and for D3's
// mass budget.  ru/rv are the big-step coupled fluxes (may be null-sized
// when want_omega == 0).
extern "C" __global__
void nest_top_column_sums(const real* __restrict__ u_pp,
                          const real* __restrict__ v_pp,
                          const real* __restrict__ u, const real* __restrict__ v,
                          const real* __restrict__ mup,
                          const real* __restrict__ mub2d,
                          const real* __restrict__ dnw,
                          const real* __restrict__ c1h, const real* __restrict__ c2h,
                          const real* __restrict__ ru, const real* __restrict__ rv,
                          int want_omega,
                          real rdx, real rdy, int klev,
                          int nz, int ny, int nx, int open_x, int open_y,
                          real* __restrict__ sums)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    size_t st = (size_t)ny * nx;
    int j = c / nx, i = c - j * nx;
    int ip = open_x ? min(i + 1, nx - 1) : (i + 1) % nx;
    int im = open_x ? max(i - 1, 0)      : (i - 1 + nx) % nx;
    int jp = open_y ? min(j + 1, ny - 1) : (j + 1) % ny;
    int jm = open_y ? max(j - 1, 0)      : (j - 1 + ny) % ny;
    int nxf = nx + 1;
    size_t cw = (size_t)j * nx + im, ce = (size_t)j * nx + ip;
    size_t cs = (size_t)jm * nx + i, cn = (size_t)jp * nx + i;
    real muw = 0.5f * ((mub2d[cw] + mup[cw]) + (mub2d[c] + mup[c]));
    real mue = 0.5f * ((mub2d[c] + mup[c]) + (mub2d[ce] + mup[ce]));
    real mus = 0.5f * ((mub2d[cs] + mup[cs]) + (mub2d[c] + mup[c]));
    real mun = 0.5f * ((mub2d[c] + mup[c]) + (mub2d[cn] + mup[cn]));
    real dmdt_pp = 0.0f, dmdt_rf = 0.0f;
    for (int k = 0; k < nz; ++k) {
        if (k == klev) {
            sums[c] = dmdt_pp;
            sums[st + c] = dmdt_rf;
        }
        size_t uw = I3S(k, j, i, ny, nxf), ue = I3S(k, j, i + 1, ny, nxf);
        size_t vs = I3S(k, j, i, ny + 1, nx), vn = I3S(k, j + 1, i, ny + 1, nx);
        real dvp = rdx * (u_pp[ue] - u_pp[uw]) + rdy * (v_pp[vn] - v_pp[vs]);
        real dvr = rdx * ((c1h[k] * mue + c2h[k]) * u[ue]
                        - (c1h[k] * muw + c2h[k]) * u[uw])
                 + rdy * ((c1h[k] * mun + c2h[k]) * v[vn]
                        - (c1h[k] * mus + c2h[k]) * v[vs]);
        dmdt_pp += dnw[k] * dvp;
        dmdt_rf += dnw[k] * dvr;
    }
    if (klev >= nz) {
        sums[c] = dmdt_pp;
        sums[st + c] = dmdt_rf;
    }
    sums[2 * st + c] = dmdt_pp;
    sums[3 * st + c] = dmdt_rf;
    if (!want_omega) return;
    real dmdt = 0.0f;
    for (int k = 0; k < nz; ++k) {
        if (k == klev) sums[4 * st + c] = dmdt;
        const int u0 = (k * ny + j) * (nx + 1) + i;
        const int v0 = (k * (ny + 1) + j) * nx + i;
        const real du = __fsub_rn(ru[u0 + 1], ru[u0]);
        const real dv = __fsub_rn(rv[v0 + nx], rv[v0]);
        const real bracket = __fadd_rn(__fmul_rn(rdx, du), __fmul_rn(rdy, dv));
        const real divv = __fmul_rn(dnw[k], bracket);
        dmdt = __fadd_rn(dmdt, divv);
    }
    if (klev >= nz) sums[4 * st + c] = dmdt;
    sums[5 * st + c] = dmdt;
}

// Upwinded flux of a half-level field q through the child's top face,
// added to its big-step tendency at the top half level (the existing
// advection kernels on the child leave that face's flux zero):
//   tend(nz-1) -= rdnw(nz-1) * F,  F = vel * q_face,
// vel the face's eta mass flux (Omega-signed: positive downward, into the
// child), q_face = q_above (the parent's value at half level k_top) for
// vel > 0, else q(nz-1) -- or, with centered = 1, the second-order face
// value fnm_top*q_above + fnp_top*q(nz-1) of the existing kernels'
// boundary-adjacent faces.  stagger 0: mass points (ny, nx), vel = ww_top;
// 1: u points (ny, nx+1), vel = 0.5*(ww_top(i-1) + ww_top(i)); 2: v points
// (ny+1, nx), vel = 0.5*(ww_top(j-1) + ww_top(j)); periodic, or clamped at
// an open boundary.  The face flux is also written to flux_out (one per
// point) for the mass/scalar budget.
extern "C" __global__
void nest_top_face_flux(real* __restrict__ tend,
                        const real* __restrict__ q,
                        const real* __restrict__ q_above,
                        const real* __restrict__ ww_top,
                        const real* __restrict__ rdnw,
                        real* __restrict__ flux_out,
                        real fnm_top, real fnp_top, int centered,
                        int stagger, int open_x, int open_y,
                        int nz, int ny, int nx)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int nys = ny + (stagger == 2 ? 1 : 0);
    int nxs = nx + (stagger == 1 ? 1 : 0);
    if (t >= nys * nxs) return;
    int j = t / nxs, i = t - j * nxs;
    real vel;
    if (stagger == 1) {
        int ia = open_x ? min(i, nx - 1) : i % nx;
        int ib = open_x ? max(i - 1, 0) : (i - 1 + nx) % nx;
        vel = 0.5f * (ww_top[(size_t)j * nx + ib] + ww_top[(size_t)j * nx + ia]);
    } else if (stagger == 2) {
        int ja = open_y ? min(j, ny - 1) : j % ny;
        int jb = open_y ? max(j - 1, 0) : (j - 1 + ny) % ny;
        vel = 0.5f * (ww_top[(size_t)jb * nx + i] + ww_top[(size_t)ja * nx + i]);
    } else {
        vel = ww_top[t];
    }
    size_t top = (size_t)(nz - 1) * nys * nxs + t;
    real qf;
    if (centered)
        qf = fnm_top * q_above[t] + fnp_top * q[top];
    else
        qf = vel > 0.0f ? q_above[t] : q[top];
    real flux = vel * qf;
    flux_out[t] = flux;
    tend[top] = tend[top] - rdnw[nz - 1] * flux;
}

// Linear interpolation in time across the parent step, one child small
// step at a time: out = a + frac*(b - a).
extern "C" __global__
void nest_top_time_interp(real* __restrict__ out,
                          const real* __restrict__ a,
                          const real* __restrict__ b,
                          real frac, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    out[t] = arn_add(a[t], arn_mul(frac, arn_sub(b[t], a[t])));
}
