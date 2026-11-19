// Tangent-linear and adjoint of woof's strict advance_uv
// (woof/core/kernels/acoustic.cu, the GPUWM_WRF_EXACT branch), which is
// WRF 4.7.1 advance_uv (dyn_em/module_small_step_em.F).  The oracle is
// WRFPLUS 4.7.1 g_advance_uv / a_advance_uv (wrftladj/module_small_step_em_
// {tl,ad}.F); see docs/dev/P3A-WRF-DRYCORE-ADJOINT.md.
//
// wreal is binary64 unless the unit is built with -DWRFPLUS_REAL=float
// (Metal, which has no binary64).  The NL statements below keep the
// primal's operation order, so the float instantiation of advance_uv_tl's
// base-state update reproduces the strict primal's words.
//
// Layout: woof's (k, j, i).  u faces (nz, ny, nx+1), v faces (nz, ny+1, nx),
// mass (nz, ny, nx), w levels (nz+1, ny, nx), 2-D (ny, nx).
//
// Active (perturbed) inputs: u_pp, v_pp, ru_t, rv_t, p_pp, p_pp_old,
// ph_pp, php, alt, al_pp, mup, mu_pp, cqu, cqv (when moist_cq), mudf (when
// use_emdiv).  Passive: phb, pb, mub2d, c1h, c2h, fnm, fnp, rdnw, msfu,
// msfv, and the scalars.

#ifndef WRFPLUS_REAL
#define WRFPLUS_REAL double
#endif
typedef WRFPLUS_REAL wreal;

#define W3(k, j, i, ny, nxs) ((size_t)(k) * (ny) * (nxs) + (size_t)(j) * (nxs) + (i))

struct UVGeom {
    int nz, ny, nx, spec_zone, base3d, top_lid, moist_cq, use_emdiv, has_msf;
    wreal rdx, rdy, dtau, smdiv, cf1, cf2, cf3, xscale, yscale;
};

// Damped p'' (acoustic.cu pdmp).
static __device__ __forceinline__
wreal pd_(const wreal* p, const wreal* pold, size_t x, wreal smdiv)
{
    return p[x] + smdiv * (p[x] - pold[x]);
}

// acoustic.cu dpn_face: linear in (p, pold), so the TL calls it on the
// perturbations.
static __device__
wreal dpn_(const wreal* p, const wreal* pold, wreal smdiv, int kf, int nz,
           size_t cA, size_t cB, size_t st, const wreal* fnm,
           const wreal* fnp, wreal cf1, wreal cf2, wreal cf3, int top_lid)
{
    if (kf == nz) {
        if (!top_lid) return (wreal)0.0;
        return (wreal)0.5 * (
            cf1 * (pd_(p, pold, cA + (size_t)(nz - 1) * st, smdiv)
                 + pd_(p, pold, cB + (size_t)(nz - 1) * st, smdiv))
          + cf2 * (pd_(p, pold, cA + (size_t)(nz - 2) * st, smdiv)
                 + pd_(p, pold, cB + (size_t)(nz - 2) * st, smdiv))
          + cf3 * (pd_(p, pold, cA + (size_t)(nz - 3) * st, smdiv)
                 + pd_(p, pold, cB + (size_t)(nz - 3) * st, smdiv)));
    }
    if (kf == 0)
        return (wreal)0.5 * (cf1 * (pd_(p, pold, cA, smdiv)
                                  + pd_(p, pold, cB, smdiv))
                           + cf2 * (pd_(p, pold, cA + st, smdiv)
                                  + pd_(p, pold, cB + st, smdiv))
                           + cf3 * (pd_(p, pold, cA + 2 * st, smdiv)
                                  + pd_(p, pold, cB + 2 * st, smdiv)));
    return (wreal)0.5 * (fnm[kf] * (pd_(p, pold, cA + (size_t)kf * st, smdiv)
                                  + pd_(p, pold, cB + (size_t)kf * st, smdiv))
                       + fnp[kf] * (pd_(p, pold, cA + (size_t)(kf - 1) * st, smdiv)
                                  + pd_(p, pold, cB + (size_t)(kf - 1) * st, smdiv)));
}

// Weight of damped p'' at mass level m in dpn_ at full level kf (one
// column's share; dpn_ averages the two columns with 0.5).
static __device__ __forceinline__
wreal dpn_weight(int kf, int m, int nz, const wreal* fnm, const wreal* fnp,
                 wreal cf1, wreal cf2, wreal cf3, int top_lid)
{
    if (kf == nz) {
        if (!top_lid) return (wreal)0.0;
        return m == nz - 1 ? cf1 : m == nz - 2 ? cf2 : m == nz - 3 ? cf3 : (wreal)0.0;
    }
    if (kf == 0)
        return m == 0 ? cf1 : m == 1 ? cf2 : m == 2 ? cf3 : (wreal)0.0;
    return m == kf ? fnm[kf] : m == kf - 1 ? fnp[kf] : (wreal)0.0;
}

// The base-state terms of one face's dpxy (acoustic.cu pgrad_face, strict).
struct FaceTerms {
    wreal aco;   // 0.5*rd*(c1h*muf+c2h)
    wreal b;     // dph + (altA+altB)*dpe + (alA+alB)*dpb
    wreal cc;    // rd*dphp
    wreal d;     // rdnw*(dpn_up-dpn_dn) - 0.5*(c1h*muB + c1h*muA)
    wreal dpe, dpb, salt;
    wreal dpxy;
};

static __device__
FaceTerms face_terms(size_t cA, size_t cB, int k, size_t st, wreal rd,
                     const wreal* ph_pp, const wreal* php, const wreal* phb,
                     const wreal* alt, const wreal* al_pp, const wreal* pb,
                     const wreal* p_pp, const wreal* p_old,
                     const wreal* mup, const wreal* mu_pp, const wreal* mub2d,
                     const wreal* c1h, const wreal* c2h, const wreal* fnm,
                     const wreal* fnp, const wreal* rdnw, const UVGeom& g)
{
    FaceTerms t;
    int nz = g.nz;
    wreal muf = (wreal)0.5 * (mup[cA] + mup[cB] + mub2d[cA] + mub2d[cB]);
    size_t a = cA + (size_t)k * st, b = cB + (size_t)k * st;
    wreal dph = (ph_pp[a + st] - ph_pp[b + st]) + (ph_pp[a] - ph_pp[b]);
    t.dpe = pd_(p_pp, p_old, a, g.smdiv) - pd_(p_pp, p_old, b, g.smdiv);
    t.dpb = g.base3d ? (pb[a] - pb[b]) : (wreal)0.0;
    t.salt = alt[a] + alt[b];
    t.aco = (wreal)0.5 * rd * (c1h[k] * muf + c2h[k]);
    t.b = dph + t.salt * t.dpe + (al_pp[a] + al_pp[b]) * t.dpb;
    wreal dpxy = t.aco * t.b;
    wreal up = dpn_(p_pp, p_old, g.smdiv, k + 1, nz, cA, cB, st, fnm, fnp,
                    g.cf1, g.cf2, g.cf3, g.top_lid);
    wreal dn = dpn_(p_pp, p_old, g.smdiv, k, nz, cA, cB, st, fnm, fnp,
                    g.cf1, g.cf2, g.cf3, g.top_lid);
    size_t ba = g.base3d ? a : (size_t)k;
    size_t bb = g.base3d ? b : (size_t)k;
    size_t bst = g.base3d ? st : 1;
    wreal half_a = (wreal)0.5 * (phb[ba] + phb[ba + bst] + php[a] + php[a + st]);
    wreal half_b = (wreal)0.5 * (phb[bb] + phb[bb + bst] + php[b] + php[b + st]);
    wreal dphp = half_a - half_b;
    t.cc = rd * dphp;
    t.d = rdnw[k] * (up - dn) - (wreal)0.5 * ((c1h[k] * mu_pp[cB]) + (c1h[k] * mu_pp[cA]));
    dpxy += t.cc * t.d;
    t.dpxy = dpxy;
    return t;
}

// TL of face_terms' dpxy at the base state t.
static __device__
wreal face_terms_tl(const FaceTerms& t, size_t cA, size_t cB, int k, size_t st,
                    wreal rd, const wreal* ph_d, const wreal* php_d,
                    const wreal* alt_d, const wreal* al_d, const wreal* p_d,
                    const wreal* pold_d, const wreal* mup_d,
                    const wreal* mupp_d, const wreal* c1h, const wreal* fnm,
                    const wreal* fnp, const wreal* rdnw, const UVGeom& g)
{
    size_t a = cA + (size_t)k * st, b = cB + (size_t)k * st;
    wreal aco_d = (wreal)0.25 * rd * c1h[k] * (mup_d[cA] + mup_d[cB]);
    wreal dph_d = (ph_d[a + st] - ph_d[b + st]) + (ph_d[a] - ph_d[b]);
    wreal dpe_d = pd_(p_d, pold_d, a, g.smdiv) - pd_(p_d, pold_d, b, g.smdiv);
    wreal b_d = dph_d + (alt_d[a] + alt_d[b]) * t.dpe + t.salt * dpe_d
              + (al_d[a] + al_d[b]) * t.dpb;
    wreal up_d = dpn_(p_d, pold_d, g.smdiv, k + 1, g.nz, cA, cB, st, fnm, fnp,
                      g.cf1, g.cf2, g.cf3, g.top_lid);
    wreal dn_d = dpn_(p_d, pold_d, g.smdiv, k, g.nz, cA, cB, st, fnm, fnp,
                      g.cf1, g.cf2, g.cf3, g.top_lid);
    wreal cc_d = rd * (wreal)0.5 * ((php_d[a] + php_d[a + st]) - (php_d[b] + php_d[b + st]));
    wreal d_d = rdnw[k] * (up_d - dn_d) - (wreal)0.5 * c1h[k] * (mupp_d[cB] + mupp_d[cA]);
        return aco_d * t.b + t.aco * b_d + cc_d * t.d + t.cc * d_d;
}

#define UV_GEOM_PARAMS                                                      \
    int nz, int ny, int nx, int spec_zone, int base3d, int top_lid,        \
    int moist_cq, int use_emdiv, int has_msf, wreal rdx, wreal rdy,        \
    wreal dtau, wreal smdiv, wreal cf1, wreal cf2, wreal cf3,              \
    wreal xscale, wreal yscale

#define UV_GEOM_INIT                                                        \
    UVGeom g; g.nz = nz; g.ny = ny; g.nx = nx; g.spec_zone = spec_zone;    \
    g.base3d = base3d; g.top_lid = top_lid; g.moist_cq = moist_cq;         \
    g.use_emdiv = use_emdiv; g.has_msf = has_msf; g.rdx = rdx; g.rdy = rdy;\
    g.dtau = dtau; g.smdiv = smdiv; g.cf1 = cf1; g.cf2 = cf2; g.cf3 = cf3;  \
    g.xscale = xscale; g.yscale = yscale

#define UV_TRAJ_PARAMS                                                      \
    const wreal* __restrict__ ru_t, const wreal* __restrict__ rv_t,        \
    const wreal* __restrict__ p_pp, const wreal* __restrict__ p_pp_old,    \
    const wreal* __restrict__ ph_pp, const wreal* __restrict__ php,        \
    const wreal* __restrict__ phb, const wreal* __restrict__ alt,          \
    const wreal* __restrict__ al_pp, const wreal* __restrict__ pb,         \
    const wreal* __restrict__ mup, const wreal* __restrict__ mu_pp,        \
    const wreal* __restrict__ mub2d, const wreal* __restrict__ c1h,        \
    const wreal* __restrict__ c2h, const wreal* __restrict__ fnm,          \
    const wreal* __restrict__ fnp, const wreal* __restrict__ rdnw,         \
    const wreal* __restrict__ cqu, const wreal* __restrict__ cqv,          \
    const wreal* __restrict__ mudf, const wreal* __restrict__ msfu,        \
    const wreal* __restrict__ msfv

#define UV_FACE_TERMS(cA, cB, k, rd)                                        \
    face_terms(cA, cB, k, st, rd, ph_pp, php, phb, alt, al_pp, pb, p_pp,   \
               p_pp_old, mup, mu_pp, mub2d, c1h, c2h, fnm, fnp, rdnw, g)

// Joint base-state and tangent-linear update (WRFPLUS g_advance_uv).
// With nl == 0 the base state u_pp/v_pp is left unchanged.
extern "C" __global__
void advance_uv_tl(wreal* __restrict__ u_pp, wreal* __restrict__ v_pp,
                   wreal* __restrict__ u_d, wreal* __restrict__ v_d,
                   UV_TRAJ_PARAMS,
                   const wreal* __restrict__ ru_d, const wreal* __restrict__ rv_d,
                   const wreal* __restrict__ p_d, const wreal* __restrict__ pold_d,
                   const wreal* __restrict__ ph_d, const wreal* __restrict__ php_d,
                   const wreal* __restrict__ alt_d, const wreal* __restrict__ al_d,
                   const wreal* __restrict__ mup_d, const wreal* __restrict__ mupp_d,
                   const wreal* __restrict__ cqu_d, const wreal* __restrict__ cqv_d,
                   const wreal* __restrict__ mudf_d,
                   int nl, UV_GEOM_PARAMS)
{
    UV_GEOM_INIT;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;
    size_t st = (size_t)ny * nx;

    if (j < ny) {
        size_t uix = W3(k, j, i, ny, nxf);
        if (j >= spec_zone && j < ny - spec_zone
            && i >= spec_zone && i <= nx - spec_zone) {
            int iA = i % nx, iB = (i - 1 + nx) % nx;
            size_t cA = (size_t)j * nx + iA, cB = (size_t)j * nx + iB;
            FaceTerms t = UV_FACE_TERMS(cA, cB, k, rdx);
            wreal dpxy_d = face_terms_tl(t, cA, cB, k, st, rdx, ph_d, php_d,
                                         alt_d, al_d, p_d, pold_d, mup_d,
                                         mupp_d, c1h, fnm, fnp, rdnw, g);
            wreal cq = moist_cq ? cqu[uix] : (wreal)1.0;
            wreal cq_d = moist_cq ? cqu_d[uix] : (wreal)0.0;
            wreal ud = u_d[uix] + dtau * ru_d[uix]
                     - dtau * (cq_d * t.dpxy + cq * dpxy_d);
            wreal u = u_pp[uix];
            u = u + dtau * ru_t[uix];
            u = u - dtau * cq * t.dpxy;
            if (use_emdiv) {
                wreal gx = xscale * (mudf[cA] - mudf[cB]);
                wreal gx_d = xscale * (mudf_d[cA] - mudf_d[cB]);
                if (has_msf) {
                    gx = gx / msfu[(size_t)j * nxf + i];
                    gx_d = gx_d / msfu[(size_t)j * nxf + i];
                }
                u = u + c1h[k] * gx;
                ud = ud + c1h[k] * gx_d;
            }
            u_d[uix] = ud;
            if (nl) u_pp[uix] = u;
        } else {
            u_d[uix] = u_d[uix] + dtau * ru_d[uix];
            if (nl) u_pp[uix] = u_pp[uix] + dtau * ru_t[uix];
        }
    }
    if (i < nx) {
        size_t vix = W3(k, j, i, nyf, nx);
        if (i >= spec_zone && i < nx - spec_zone
            && j >= spec_zone && j <= ny - spec_zone) {
            int jA = j % ny, jB = (j - 1 + ny) % ny;
            size_t cA = (size_t)jA * nx + i, cB = (size_t)jB * nx + i;
            FaceTerms t = UV_FACE_TERMS(cA, cB, k, rdy);
            wreal dpxy_d = face_terms_tl(t, cA, cB, k, st, rdy, ph_d, php_d,
                                         alt_d, al_d, p_d, pold_d, mup_d,
                                         mupp_d, c1h, fnm, fnp, rdnw, g);
            wreal cq = moist_cq ? cqv[vix] : (wreal)1.0;
            wreal cq_d = moist_cq ? cqv_d[vix] : (wreal)0.0;
            wreal vd = v_d[vix] + dtau * rv_d[vix]
                     - dtau * (cq_d * t.dpxy + cq * dpxy_d);
            wreal v = v_pp[vix];
            v = v + dtau * rv_t[vix];
            v = v - dtau * cq * t.dpxy;
            if (use_emdiv) {
                wreal gy = yscale * (mudf[cA] - mudf[cB]);
                wreal gy_d = yscale * (mudf_d[cA] - mudf_d[cB]);
                if (has_msf) {
                    gy = gy * msfv[(size_t)j * nx + i];
                    gy_d = gy_d * msfv[(size_t)j * nx + i];
                }
                v = v + c1h[k] * gy;
                vd = vd + c1h[k] * gy_d;
            }
            v_d[vix] = vd;
            if (nl) v_pp[vix] = v;
        } else {
            v_d[vix] = v_d[vix] + dtau * rv_d[vix];
            if (nl) v_pp[vix] = v_pp[vix] + dtau * rv_t[vix];
        }
    }
}

// Adjoint, pass 1 of 3: one thread per face.  Pointwise adjoints of ru_t,
// rv_t, cqu, cqv, and the face seeds the gather passes read:
//   s_b  = g*aco        (adjoint of b)
//   s_cc = g*d          (adjoint of cc = rd*dphp)
//   s_d  = g*cc         (adjoint of d)
//   s_m  = g*b*0.5*rd*c1h (adjoint of muf)
//   s_e  = c1h*scale(/msf)*ua (adjoint of mudf difference)
// with g = -dtau*cq*ua the adjoint of dpxy.  u_a/v_a are unchanged (the
// update is u_out = u_in + ...).
extern "C" __global__
void advance_uv_ad_face(const wreal* __restrict__ u_a, const wreal* __restrict__ v_a,
                        UV_TRAJ_PARAMS,
                        wreal* __restrict__ ru_a, wreal* __restrict__ rv_a,
                        wreal* __restrict__ cqu_a, wreal* __restrict__ cqv_a,
                        wreal* __restrict__ su, wreal* __restrict__ sv,
                        UV_GEOM_PARAMS)
{
    UV_GEOM_INIT;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;
    size_t st = (size_t)ny * nx;
    size_t nu = (size_t)nz * ny * nxf, nv = (size_t)nz * nyf * nx;

    if (j < ny) {
        size_t uix = W3(k, j, i, ny, nxf);
        wreal ua = u_a[uix];
        ru_a[uix] += dtau * ua;
        wreal sb = 0, scc = 0, sd = 0, sm = 0, se = 0;
        if (j >= spec_zone && j < ny - spec_zone
            && i >= spec_zone && i <= nx - spec_zone) {
            int iA = i % nx, iB = (i - 1 + nx) % nx;
            size_t cA = (size_t)j * nx + iA, cB = (size_t)j * nx + iB;
            FaceTerms t = UV_FACE_TERMS(cA, cB, k, rdx);
            wreal cq = moist_cq ? cqu[uix] : (wreal)1.0;
            if (moist_cq) cqu_a[uix] += -dtau * t.dpxy * ua;
            wreal gg = -dtau * cq * ua;
            sb = gg * t.aco;
            scc = gg * t.d;
            sd = gg * t.cc;
            sm = gg * t.b * (wreal)0.5 * rdx * c1h[k];
            if (use_emdiv) {
                se = c1h[k] * xscale * ua;
                if (has_msf) se = se / msfu[(size_t)j * nxf + i];
            }
        }
        su[uix] = sb; su[nu + uix] = scc; su[2 * nu + uix] = sd;
        su[3 * nu + uix] = sm; su[4 * nu + uix] = se;
    }
    if (i < nx) {
        size_t vix = W3(k, j, i, nyf, nx);
        wreal va = v_a[vix];
        rv_a[vix] += dtau * va;
        wreal sb = 0, scc = 0, sd = 0, sm = 0, se = 0;
        if (i >= spec_zone && i < nx - spec_zone
            && j >= spec_zone && j <= ny - spec_zone) {
            int jA = j % ny, jB = (j - 1 + ny) % ny;
            size_t cA = (size_t)jA * nx + i, cB = (size_t)jB * nx + i;
            FaceTerms t = UV_FACE_TERMS(cA, cB, k, rdy);
            wreal cq = moist_cq ? cqv[vix] : (wreal)1.0;
            if (moist_cq) cqv_a[vix] += -dtau * t.dpxy * va;
            wreal gg = -dtau * cq * va;
            sb = gg * t.aco;
            scc = gg * t.d;
            sd = gg * t.cc;
            sm = gg * t.b * (wreal)0.5 * rdy * c1h[k];
            if (use_emdiv) {
                se = c1h[k] * yscale * va;
                if (has_msf) se = se * msfv[(size_t)j * nx + i];
            }
        }
        sv[vix] = sb; sv[nv + vix] = scc; sv[2 * nv + vix] = sd;
        sv[3 * nv + vix] = sm; sv[4 * nv + vix] = se;
    }
}

// A face that reads mass column c: its seed offset and the sign of c in
// the face's differences (+1 for cA, -1 for cB).
struct FaceRef { size_t base; wreal sign; int isv; };

// Up to 4 u faces and 4 v faces touch a column; under periodic wrap the
// same face can reach c from both sides (nx == 1), so each (face, side)
// pair is listed separately and the list is deduplicated by face index.
static __device__
int column_faces(int j, int i, const UVGeom& g, FaceRef* out)
{
    int n = 0, nx = g.nx, ny = g.ny, sz = g.spec_zone;
    int cand[4] = {i, i + 1, i == 0 ? nx : -1, i == nx - 1 ? 0 : -1};
    for (int q = 0; q < 4; ++q) {
        int fi = cand[q];
        if (fi < 0) continue;
        int dup = 0;
        for (int p = 0; p < q; ++p) dup |= (cand[p] == fi);
        if (dup) continue;
        if (!(j >= sz && j < ny - sz && fi >= sz && fi <= nx - sz)) continue;
        int iA = fi % nx, iB = (fi - 1 + nx) % nx;
        size_t base = (size_t)j * (nx + 1) + fi;
        if (iA == i) { out[n].base = base; out[n].sign = (wreal)1.0; out[n].isv = 0; ++n; }
        if (iB == i) { out[n].base = base; out[n].sign = (wreal)-1.0; out[n].isv = 0; ++n; }
    }
    int candj[4] = {j, j + 1, j == 0 ? ny : -1, j == ny - 1 ? 0 : -1};
    for (int q = 0; q < 4; ++q) {
        int fj = candj[q];
        if (fj < 0) continue;
        int dup = 0;
        for (int p = 0; p < q; ++p) dup |= (candj[p] == fj);
        if (dup) continue;
        if (!(i >= sz && i < nx - sz && fj >= sz && fj <= ny - sz)) continue;
        int jA = fj % ny, jB = (fj - 1 + ny) % ny;
        size_t base = (size_t)fj * nx + i;
        if (jA == j) { out[n].base = base; out[n].sign = (wreal)1.0; out[n].isv = 1; ++n; }
        if (jB == j) { out[n].base = base; out[n].sign = (wreal)-1.0; out[n].isv = 1; ++n; }
    }
    return n;
}

// The other column of a face (to rebuild dpe, salt, dpb in the gather).
static __device__ __forceinline__
size_t other_column(const FaceRef& f, const UVGeom& g)
{
    int nx = g.nx, ny = g.ny;
    if (f.isv == 0) {
        int fi = (int)(f.base % (size_t)(nx + 1)), j = (int)(f.base / (size_t)(nx + 1));
        int iA = fi % nx, iB = (fi - 1 + nx) % nx;
        return (size_t)j * nx + (f.sign > 0 ? iB : iA);
    }
    int i = (int)(f.base % (size_t)nx), fj = (int)(f.base / (size_t)nx);
    int jA = fj % ny, jB = (fj - 1 + ny) % ny;
    return (size_t)(f.sign > 0 ? jB : jA) * nx + i;
}

// Adjoint, pass 2 of 3: one thread per (level, column) over w levels
// 0..nz.  Gathers ph_pp and php at w level k, and alt, al_pp, p_pp and
// p_pp_old at mass level k (k < nz), from the face seeds.  Deterministic:
// each output word is written by one thread in a fixed order.
extern "C" __global__
void advance_uv_ad_gather(UV_TRAJ_PARAMS,
                          const wreal* __restrict__ su, const wreal* __restrict__ sv,
                          wreal* __restrict__ ph_a, wreal* __restrict__ php_a,
                          wreal* __restrict__ alt_a, wreal* __restrict__ al_a,
                          wreal* __restrict__ p_a, wreal* __restrict__ pold_a,
                          UV_GEOM_PARAMS)
{
    UV_GEOM_INIT;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (nz + 1) * ny * nx) return;
    int k = tid / (ny * nx);
    int c = tid - k * ny * nx;
    int j = c / nx, i = c - j * nx;
    size_t st = (size_t)ny * nx;
    size_t nu = (size_t)nz * ny * (nx + 1), nv = (size_t)nz * (ny + 1) * nx;
    size_t lu = (size_t)ny * (nx + 1), lv = (size_t)(ny + 1) * nx;
    FaceRef f[16];
    int nf = column_faces(j, i, g, f);

    // w level k: ph_pp and php enter faces at levels k (as [a]) and k-1
    // (as [a+st]).
    wreal aph = 0, aphp = 0;
    for (int q = 0; q < nf; ++q) {
        const wreal* s = f[q].isv ? sv : su;
        size_t ns = f[q].isv ? nv : nu, ls = f[q].isv ? lv : lu;
        for (int kk = k - 1; kk <= k; ++kk) {
            if (kk < 0 || kk >= nz) continue;
            size_t x = (size_t)kk * ls + f[q].base;
            aph += f[q].sign * s[x];
            aphp += f[q].sign * (wreal)0.5 * (f[q].isv ? rdy : rdx) * s[ns + x];
        }
    }
    ph_a[tid] += aph;
    php_a[tid] += aphp;
    if (k >= nz) return;

    // mass level k.
    size_t a = (size_t)k * st + c;
    wreal aalt = 0, aal = 0, aP = 0;
    for (int q = 0; q < nf; ++q) {
        const wreal* s = f[q].isv ? sv : su;
        size_t ns = f[q].isv ? nv : nu, ls = f[q].isv ? lv : lu;
        size_t x = (size_t)k * ls + f[q].base;
        size_t o = other_column(f[q], g);
        size_t b = (size_t)k * st + o;
        // Rebuild the face's dpe/dpb with this column's sign convention.
        size_t fa = f[q].sign > 0 ? a : b, fb = f[q].sign > 0 ? b : a;
        wreal dpe = pd_(p_pp, p_pp_old, fa, smdiv) - pd_(p_pp, p_pp_old, fb, smdiv);
        wreal dpb = base3d ? (pb[fa] - pb[fb]) : (wreal)0.0;
        wreal sb = s[x];
        aalt += sb * dpe;
        aal += sb * dpb;
        aP += f[q].sign * sb * (alt[a] + alt[b]);
        // dpn: level m = k of this column enters dpn at full levels kf;
        // dpn(kf) is read by faces kf-1 (as up, +rdnw) and kf (as dn, -rdnw).
        int kfs[4] = {k, k + 1, 0, nz};
        for (int q2 = 0; q2 < 4; ++q2) {
            int kf = kfs[q2];
            int dup = 0;
            for (int p2 = 0; p2 < q2; ++p2) dup |= (kfs[p2] == kf);
            if (dup || kf < 0 || kf > nz) continue;
            wreal w = dpn_weight(kf, k, nz, fnm, fnp, cf1, cf2, cf3, top_lid);
            if (w == (wreal)0.0) continue;
            wreal adpn = 0;
            if (kf >= 1) adpn += rdnw[kf - 1] * s[2 * ns + (size_t)(kf - 1) * ls + f[q].base];
            if (kf <= nz - 1) adpn -= rdnw[kf] * s[2 * ns + (size_t)kf * ls + f[q].base];
            aP += (wreal)0.5 * w * adpn;
        }
    }
    alt_a[a] += aalt;
    al_a[a] += aal;
    p_a[a] += aP + smdiv * aP;
    pold_a[a] += -smdiv * aP;
}

// Adjoint, pass 3 of 3: one thread per column.  mup, mu_pp and mudf.
extern "C" __global__
void advance_uv_ad_column(const wreal* __restrict__ c1h,
                          const wreal* __restrict__ su, const wreal* __restrict__ sv,
                          wreal* __restrict__ mup_a, wreal* __restrict__ mupp_a,
                          wreal* __restrict__ mudf_a, UV_GEOM_PARAMS)
{
    UV_GEOM_INIT;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    int j = c / nx, i = c - j * nx;
    size_t nu = (size_t)nz * ny * (nx + 1), nv = (size_t)nz * (ny + 1) * nx;
    size_t lu = (size_t)ny * (nx + 1), lv = (size_t)(ny + 1) * nx;
    FaceRef f[16];
    int nf = column_faces(j, i, g, f);
    wreal am = 0, amu = 0, amd = 0;
    for (int q = 0; q < nf; ++q) {
        const wreal* s = f[q].isv ? sv : su;
        size_t ns = f[q].isv ? nv : nu, ls = f[q].isv ? lv : lu;
        for (int k = 0; k < nz; ++k) {
            size_t x = (size_t)k * ls + f[q].base;
            am += (wreal)0.5 * s[3 * ns + x];
            amu += (wreal)-0.5 * c1h[k] * s[2 * ns + x];
            amd += f[q].sign * s[4 * ns + x];
        }
    }
    mup_a[c] += am;
    mupp_a[c] += amu;
    if (use_emdiv) mudf_a[c] += amd;
}
