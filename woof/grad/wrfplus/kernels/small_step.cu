// Tangent-linear and adjoint of woof's strict small_step_init_uv/_column
// (WRF small_step_prep, then calc_p_rho at step 0) and
// small_step_finish_uv/_column (WRF small_step_finish, whose muus/muvs
// are calc_mu_uv_1 of WRF's muts), all in woof/core/kernels/dycore.cu
// under GPUWM_WRF_EXACT.  The oracle is WRFPLUS 4.7.1 g_/a_small_step_prep,
// g_/a_calc_p_rho and G_/A_SMALL_STEP_FINISH
// (wrftladj/module_small_step_em_{tl,ad}.F); see
// docs/dev/P3A-WRF-DRYCORE-ADJOINT.md.
//
// wreal is binary64 unless the unit is built with -DWRFPLUS_REAL=float.
// The base-state statements keep the strict primal's operation order, so
// the float instantiation of the TL's base state gives the primal's words.
//
// Layout (k, j, i).  u-type (nz, ny, nx+1), v-type (nz, ny+1, nx), half
// levels (nz, ny, nx), w levels (nz+1, ny, nx), columns (ny, nx).
//
// Face masses: a u face i sits between cells b = i-1 and a = i; on a
// non-periodic axis the boundary faces clamp both to the edge cell, on a
// periodic one they wrap (calc_mu_uv).  The adjoints of the face masses go
// to the cells through a gather: one thread per face column sums the
// mass adjoints over k into scratch, then one thread per cell adds those
// of every face that names it.  Every output word has one writer.

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

#define WGAMMA ((wreal)1.4)

#define SI3(k, j, i, nyy, nxx) \
    ((size_t)(k) * (size_t)(nyy) * (size_t)(nxx) + (size_t)(j) * (size_t)(nxx) + (size_t)(i))

#define SS_GEOM int nz, int ny, int nx, int has_msf, int boundary_x,        \
    int boundary_y, int rk_step, int remove_hdiab, wreal hdiab_dt

// Cells a (right/north) and b (left/south) of face f on an n-cell axis.
static __device__ __forceinline__ int ss_cell_a(int f, int n, int bnd)
{
    return bnd ? min(f, n - 1) : f % n;
}
static __device__ __forceinline__ int ss_cell_b(int f, int n, int bnd)
{
    return bnd ? max(f - 1, 0) : (f - 1 + n) % n;
}
// Face q (0-3) whose stencil names cell p, one entry per (face, role):
// q0 face p (p is its a), q1 face p+1 (p is its b); q2 face 0 as b and q3
// face n as a, where the clamp (non-periodic) or the wrap (periodic) also
// names p.  -1: none.
static __device__ __forceinline__ int ss_face(int q, int p, int n, int bnd)
{
    if (q == 0) return p;
    if (q == 1) return p + 1;
    if (q == 2) return (bnd ? p == 0 : p == n - 1) ? 0 : -1;
    return (bnd ? p == n - 1 : p == 0) ? n : -1;
}

// ---------------------------------------------------------------- init

#define SI_STATE(s)                                                         \
    wreal* __restrict__ u_pp##s, wreal* __restrict__ v_pp##s,              \
    wreal* __restrict__ w_pp##s, wreal* __restrict__ th_pp##s,             \
    wreal* __restrict__ ph_pp##s, wreal* __restrict__ mu_pp##s,            \
    wreal* __restrict__ al_pp##s, wreal* __restrict__ p_pp##s,             \
    wreal* __restrict__ p_pp_old##s

#define SI_TRAJ(s)                                                          \
    const wreal* __restrict__ u0##s, const wreal* __restrict__ v0##s,      \
    const wreal* __restrict__ u##s, const wreal* __restrict__ v##s,        \
    const wreal* __restrict__ w0##s, const wreal* __restrict__ w##s,       \
    const wreal* __restrict__ thp0##s, const wreal* __restrict__ thp##s,   \
    const wreal* __restrict__ php0##s, const wreal* __restrict__ php##s,   \
    const wreal* __restrict__ p##s, const wreal* __restrict__ alt##s,      \
    const wreal* __restrict__ mup0##s, const wreal* __restrict__ mup##s

#define SI_ADJ                                                              \
    wreal* __restrict__ u0_a, wreal* __restrict__ v0_a,                    \
    wreal* __restrict__ u_a, wreal* __restrict__ v_a,                      \
    wreal* __restrict__ w0_a, wreal* __restrict__ w_a,                     \
    wreal* __restrict__ thp0_a, wreal* __restrict__ thp_a,                 \
    wreal* __restrict__ php0_a, wreal* __restrict__ php_a,                 \
    wreal* __restrict__ p_a, wreal* __restrict__ alt_a,                    \
    wreal* __restrict__ mup0_a, wreal* __restrict__ mup_a

#define SI_PASSIVE                                                          \
    const wreal* __restrict__ mub2d, const wreal* __restrict__ c1h,        \
    const wreal* __restrict__ c2h, const wreal* __restrict__ c1f,          \
    const wreal* __restrict__ c2f, const wreal* __restrict__ rdnw,         \
    const wreal* __restrict__ msfu, const wreal* __restrict__ msfv,        \
    const wreal* __restrict__ msft

#define SS_SCRATCH wreal* __restrict__ xm1, wreal* __restrict__ xm2,        \
    wreal* __restrict__ ym1, wreal* __restrict__ ym2

// Strict face masses: msf from mup, mtf from mup0 (mup on rk_step 1).
static __device__ __forceinline__ void si_face_mass(
    const wreal* mup0, const wreal* mup, const wreal* mub2d, size_t ca,
    size_t cb, int rk_step, wreal* msf, wreal* mtf)
{
    *msf = (wreal)0.5 * (((mup[ca] + mup[cb]) + mub2d[ca]) + mub2d[cb]);
    *mtf = rk_step == 1 ? *msf
         : (wreal)0.5 * (((mub2d[ca] + mup0[ca]) + mub2d[cb]) + mup0[cb]);
}

extern "C" __global__
void small_step_init_uv_tl(SI_STATE(), SI_STATE(_d), SI_TRAJ(), SI_TRAJ(_d),
                           SI_PASSIVE, int nl, SS_GEOM)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;
    if (j < ny) {
        size_t ca = (size_t)j * nx + ss_cell_a(i, nx, boundary_x);
        size_t cb = (size_t)j * nx + ss_cell_b(i, nx, boundary_x);
        wreal msf, mtf;
        si_face_mass(mup0, mup, mub2d, ca, cb, rk_step, &msf, &mtf);
        wreal msf_d = (wreal)0.5 * (mup_d[ca] + mup_d[cb]);
        wreal mtf_d = rk_step == 1 ? msf_d : (wreal)0.5 * (mup0_d[ca] + mup0_d[cb]);
        wreal ct = c1h[k] * mtf + c2h[k];
        wreal cs = c1h[k] * msf + c2h[k];
        size_t ix = SI3(k, j, i, ny, nxf);
        wreal value = ct * u0[ix] - cs * u[ix];
        wreal vd = c1h[k] * mtf_d * u0[ix] + ct * u0_d[ix]
                 - c1h[k] * msf_d * u[ix] - cs * u_d[ix];
        if (has_msf) {
            value = value / msfu[(size_t)j * nxf + i];
            vd = vd / msfu[(size_t)j * nxf + i];
        }
        if (nl) u_pp[ix] = value;
        if (WRFPLUS_TL) u_pp_d[ix] = vd;
    }
    if (i < nx) {
        size_t ca = (size_t)ss_cell_a(j, ny, boundary_y) * nx + i;
        size_t cb = (size_t)ss_cell_b(j, ny, boundary_y) * nx + i;
        wreal msf, mtf;
        si_face_mass(mup0, mup, mub2d, ca, cb, rk_step, &msf, &mtf);
        wreal msf_d = (wreal)0.5 * (mup_d[ca] + mup_d[cb]);
        wreal mtf_d = rk_step == 1 ? msf_d : (wreal)0.5 * (mup0_d[ca] + mup0_d[cb]);
        wreal ct = c1h[k] * mtf + c2h[k];
        wreal cs = c1h[k] * msf + c2h[k];
        size_t ix = SI3(k, j, i, nyf, nx);
        wreal value = ct * v0[ix] - cs * v[ix];
        wreal vd = c1h[k] * mtf_d * v0[ix] + ct * v0_d[ix]
                 - c1h[k] * msf_d * v[ix] - cs * v_d[ix];
        if (has_msf) {                       // strict: msfv is MF_VX_INV
            value = value * msfv[(size_t)j * nx + i];
            vd = vd * msfv[(size_t)j * nx + i];
        }
        if (nl) v_pp[ix] = value;
        if (WRFPLUS_TL) v_pp_d[ix] = vd;
    }
}

extern "C" __global__
void small_step_init_column_tl(SI_STATE(), SI_STATE(_d), SI_TRAJ(), SI_TRAJ(_d),
                               SI_PASSIVE, int nl, SS_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    size_t st = (size_t)ny * nx;
    wreal mut = mub2d[c] + mup0[c];
    wreal mus = mub2d[c] + mup[c];
    wreal mupp = mup0[c] - mup[c];
    wreal mut_d = mup0_d[c], mus_d = mup_d[c];
    wreal mupp_d = mup0_d[c] - mup_d[c];
    if (nl) mu_pp[c] = mupp;
    if (WRFPLUS_TL) mu_pp_d[c] = mupp_d;
    for (int k = 0; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        wreal cft = c1f[k] * mut + c2f[k];
        wreal cfs = c1f[k] * mus + c2f[k];
        wreal value = cft * w0[f] - cfs * w[f];
        wreal vd = c1f[k] * mut_d * w0[f] + cft * w0_d[f]
                 - c1f[k] * mus_d * w[f] - cfs * w_d[f];
        if (has_msf) { value = value / msft[c]; vd = vd / msft[c]; }
        if (nl) { w_pp[f] = value; ph_pp[f] = php0[f] - php[f]; }
        if (WRFPLUS_TL) w_pp_d[f] = vd;
        if (WRFPLUS_TL) ph_pp_d[f] = php0_d[f] - php_d[f];
    }
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        wreal cht = c1h[k] * mut + c2h[k];
        wreal chs = c1h[k] * mus + c2h[k];
        wreal cht_d = c1h[k] * mut_d, chs_d = c1h[k] * mus_d;
        wreal tht = thp0[h], ths = thp[h];
        wreal thpp = cht * tht - chs * ths;
        wreal thpp_d = cht_d * tht + cht * thp0_d[h] - chs_d * ths - chs * thp_d[h];
        wreal c2a = WGAMMA * p[h] / alt[h];
        wreal c2a_d = WGAMMA * (p_d[h] - p[h] / alt[h] * alt_d[h]) / alt[h];
        wreal al0 = c1h[k] * mupp;
        wreal al1 = alt[h] * al0;
        wreal dph = (php0[h + st] - php[h + st]) - (php0[h] - php[h]);
        wreal dph_d = (php0_d[h + st] - php_d[h + st]) - (php0_d[h] - php_d[h]);
        wreal al2 = rdnw[k] * dph;
        wreal S = al1 + al2;
        wreal al = (-(wreal)1.0 / cht) * S;
        wreal S_d = alt_d[h] * al0 + alt[h] * c1h[k] * mupp_d + rdnw[k] * dph_d;
        wreal al_d = (-(wreal)1.0 / cht) * S_d + S / (cht * cht) * cht_d;
        wreal p1 = al0 * ths;
        wreal p2 = thpp - p1;
        wreal p3 = alt[h] * p2;
        wreal p4 = cht * ((wreal)300.0 + ths);
        wreal p5 = p3 / p4;
        wreal p6 = p5 - al;
        wreal pnew = c2a * p6;
        wreal p1_d = c1h[k] * (mupp_d * ths + mupp * thp_d[h]);
        wreal p2_d = thpp_d - p1_d;
        wreal p3_d = alt_d[h] * p2 + alt[h] * p2_d;
        wreal p4_d = cht_d * ((wreal)300.0 + ths) + cht * thp_d[h];
        wreal p5_d = (p3_d - p5 * p4_d) / p4;
        wreal p6_d = p5_d - al_d;
        wreal pnew_d = c2a_d * p6 + c2a * p6_d;
        if (nl) {
            th_pp[h] = thpp; al_pp[h] = al; p_pp[h] = pnew; p_pp_old[h] = pnew;
        }
        if (WRFPLUS_TL) { th_pp_d[h] = thpp_d; al_pp_d[h] = al_d; p_pp_d[h] = pnew_d; }
        if (WRFPLUS_TL) p_pp_old_d[h] = pnew_d;
    }
}

// The adjoints run one thread per (level chunk, column or face): chunk ck
// covers levels [ck*SS_KCHUNK, min(nz, (ck+1)*SS_KCHUNK)), the last chunk
// also the top w level nz.  Column sums (the mass adjoints) leave the
// chunk as partials in scratch (c1, c2; xm*/ym* per face), and the
// adjoint of ph'' at a chunk's bottom level misses the layer below's
// share, which the chunk below leaves in ``edge``; the cell gather adds
// them all.  Every word has one writer.
#ifndef SS_KCHUNK
#define SS_KCHUNK 8
#endif

#define SS_CHUNK_SCRATCH                                                    \
    wreal* __restrict__ c1, wreal* __restrict__ c2, wreal* __restrict__ edge

// Chunked column adjoint of small_step_init_column: consumes every
// output's adjoint and adds into the column's own inputs; the mup/mup0
// sums go to c1/c2 (chunk partials).
extern "C" __global__
void small_step_init_column_ad(SI_STATE(_a), SI_TRAJ(), SI_ADJ, SS_CHUNK_SCRATCH,
                               SI_PASSIVE, SS_GEOM)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    int nck = (nz + SS_KCHUNK - 1) / SS_KCHUNK;
    if (tid >= nck * ny * nx) return;
    int ck = tid / (ny * nx);
    int c = tid - ck * ny * nx;
    int k0 = ck * SS_KCHUNK;
    int k1 = min(nz, k0 + SS_KCHUNK);
    int last = k1 == nz;
    wreal mut = mub2d[c] + mup0[c];
    wreal mus = mub2d[c] + mup[c];
    wreal mupp = mup0[c] - mup[c];
    wreal a_mut = 0, a_mus = 0, a_mupp = 0;
    wreal aph_prev = 0;    // ph'' adjoint at level k from layer k-1 (in chunk)
    for (int k = k0; k < k1; ++k) {
        size_t h = (size_t)k * st + c;
        wreal cht = c1h[k] * mut + c2h[k];
        wreal chs = c1h[k] * mus + c2h[k];
        wreal tht = thp0[h], ths = thp[h];
        wreal thpp = cht * tht - chs * ths;
        // The recompute takes reciprocals: these words feed only the
        // adjoint, which no test holds to the primal's bits.
        wreal ralt = (wreal)1.0 / alt[h];
        wreal c2a = WGAMMA * p[h] * ralt;
        wreal al0 = c1h[k] * mupp;
        wreal al1 = alt[h] * al0;
        wreal dph = (php0[h + st] - php[h + st]) - (php0[h] - php[h]);
        wreal S = al1 + rdnw[k] * dph;
        wreal p1 = al0 * ths;
        wreal p2 = thpp - p1;
        wreal p4 = cht * ((wreal)300.0 + ths);
        wreal rp4 = (wreal)1.0 / p4;
        wreal rcht = (wreal)1.0 / cht;
        wreal p5 = alt[h] * p2 * rp4;

        wreal P = p_pp_a[h] + p_pp_old_a[h];
        p_pp_a[h] = 0; p_pp_old_a[h] = 0;
        wreal a_c2a = P * (p5 + rcht * S);
        wreal a_p6 = P * c2a;
        wreal a_al = al_pp_a[h] - a_p6;
        al_pp_a[h] = 0;
        wreal a_p3 = a_p6 * rp4;
        wreal a_p4 = -a_p3 * p5;
        wreal a_cht = a_p4 * ((wreal)300.0 + ths);
        wreal a_ths = a_p4 * cht;
        wreal a_alt = a_p3 * p2;
        wreal a_p2 = a_p3 * alt[h];
        wreal a_thpp = a_p2 + th_pp_a[h];
        th_pp_a[h] = 0;
        // p1 = c1h*mupp*ths
        a_mupp -= a_p2 * c1h[k] * ths;
        a_ths -= a_p2 * al0;
        // al = (-1/cht)*S
        wreal a_S = -a_al * rcht;
        a_cht += a_al * S * rcht * rcht;
        a_alt += a_S * al0;
        a_mupp += a_S * alt[h] * c1h[k];
        wreal a_dph = a_S * rdnw[k];
        // thpp = cht*tht - chs*ths
        a_cht += a_thpp * tht;
        thp0_a[h] += a_thpp * cht;
        wreal a_chs = -a_thpp * ths;
        a_ths -= a_thpp * chs;
        thp_a[h] += a_ths;
        // c2a = gamma*p/alt
        p_a[h] += a_c2a * WGAMMA * ralt;
        a_alt -= a_c2a * c2a * ralt;
        alt_a[h] += a_alt;
        a_mut += a_cht * c1h[k];
        a_mus += a_chs * c1h[k];
        // dph = ph''[k+1] - ph''[k], ph'' = php0 - php.
        wreal aph_k = ph_pp_a[h] + aph_prev - a_dph;
        ph_pp_a[h] = 0;
        php0_a[h] += aph_k; php_a[h] -= aph_k;
        aph_prev = a_dph;
    }
    if (last) {
        size_t f = (size_t)nz * st + c;
        wreal aph_k = ph_pp_a[f] + aph_prev;
        ph_pp_a[f] = 0;
        php0_a[f] += aph_k; php_a[f] -= aph_k;
    } else {
        edge[(size_t)ck * st + c] = aph_prev;     // to level k1, in the gather
    }
    for (int k = k0; k < (last ? nz + 1 : k1); ++k) {
        size_t f = (size_t)k * st + c;
        wreal cft = c1f[k] * mut + c2f[k];
        wreal cfs = c1f[k] * mus + c2f[k];
        wreal av = w_pp_a[f];
        w_pp_a[f] = 0;
        if (has_msf) av = av / msft[c];
        a_mut += av * w0[f] * c1f[k];
        w0_a[f] += av * cft;
        a_mus -= av * w[f] * c1f[k];
        w_a[f] -= av * cfs;
    }
    if (ck == 0) {
        a_mupp += mu_pp_a[c];
        mu_pp_a[c] = 0;
    }
    c1[(size_t)ck * st + c] = a_mus - a_mupp;     // -> mup
    c2[(size_t)ck * st + c] = a_mupp + a_mut;     // -> mup0
}

// Chunked face pass of small_step_init_uv's adjoint: consumes u_pp/v_pp's
// adjoints, adds into u0/u/v0/v, and leaves each face's chunk sums of the
// msf (xm1/ym1) and mtf (xm2/ym2) adjoints for the cell gather.
extern "C" __global__
void small_step_init_uv_ad(SI_STATE(_a), SI_TRAJ(), SI_ADJ, SS_SCRATCH,
                           SI_PASSIVE, SS_GEOM)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    int nck = (nz + SS_KCHUNK - 1) / SS_KCHUNK;
    if (tid >= nck * nyf * nxf) return;
    int ck = tid / (nyf * nxf);
    int r = tid - ck * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;
    int k0 = ck * SS_KCHUNK;
    int k1 = min(nz, k0 + SS_KCHUNK);
    if (j < ny) {
        wreal ams = 0, amt = 0;
        size_t ca = (size_t)j * nx + ss_cell_a(i, nx, boundary_x);
        size_t cb = (size_t)j * nx + ss_cell_b(i, nx, boundary_x);
        wreal msf, mtf;
        si_face_mass(mup0, mup, mub2d, ca, cb, rk_step, &msf, &mtf);
        for (int k = k0; k < k1; ++k) {
            size_t ix = SI3(k, j, i, ny, nxf);
            wreal av = u_pp_a[ix];
            if (has_msf) av = av / msfu[(size_t)j * nxf + i];
            u_pp_a[ix] = 0;
            wreal ct = c1h[k] * mtf + c2h[k];
            wreal cs = c1h[k] * msf + c2h[k];
            u0_a[ix] += av * ct;
            u_a[ix] -= av * cs;
            amt += av * u0[ix] * c1h[k];
            ams -= av * u[ix] * c1h[k];
        }
        if (rk_step == 1) { ams += amt; amt = 0; }
        size_t o = (size_t)ck * ny * nxf + (size_t)j * nxf + i;
        xm1[o] = ams;
        xm2[o] = amt;
    }
    if (i < nx) {
        wreal ams = 0, amt = 0;
        size_t ca = (size_t)ss_cell_a(j, ny, boundary_y) * nx + i;
        size_t cb = (size_t)ss_cell_b(j, ny, boundary_y) * nx + i;
        wreal msf, mtf;
        si_face_mass(mup0, mup, mub2d, ca, cb, rk_step, &msf, &mtf);
        for (int k = k0; k < k1; ++k) {
            size_t ix = SI3(k, j, i, nyf, nx);
            wreal av = v_pp_a[ix];
            if (has_msf) av = av * msfv[(size_t)j * nx + i];
            v_pp_a[ix] = 0;
            wreal ct = c1h[k] * mtf + c2h[k];
            wreal cs = c1h[k] * msf + c2h[k];
            v0_a[ix] += av * ct;
            v_a[ix] -= av * cs;
            amt += av * v0[ix] * c1h[k];
            ams -= av * v[ix] * c1h[k];
        }
        if (rk_step == 1) { ams += amt; amt = 0; }
        size_t o = (size_t)ck * nyf * nx + (size_t)j * nx + i;
        ym1[o] = ams;
        ym2[o] = amt;
    }
}

// Cell gather.  A face mass is the mean of cells a and b, so each cell
// takes half of every face's adjoint that names it (twice on a clamped
// boundary face, where a = b): m1 adds the xm1/ym1 chunk sums and the
// c1 column partials, m2 the xm2/ym2 and c2 ones.  ``finish`` first adds
// m1's incoming value (mup's output adjoint) into ``pre`` (mu_pp's);
// otherwise (init) the chunk-boundary ph'' adjoints in ``edge`` go to
// php0/php at each chunk's bottom level.
extern "C" __global__
void small_step_mass_gather(wreal* __restrict__ m1, wreal* __restrict__ m2,
                            wreal* __restrict__ pre, wreal* __restrict__ ph0_a,
                            wreal* __restrict__ ph_a,
                            const wreal* __restrict__ xm1, const wreal* __restrict__ xm2,
                            const wreal* __restrict__ ym1, const wreal* __restrict__ ym2,
                            const wreal* __restrict__ c1, const wreal* __restrict__ c2,
                            const wreal* __restrict__ edge, int finish, SS_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    int j = c / nx, i = c - j * nx;
    int nyf = ny + 1, nxf = nx + 1;
    size_t st = (size_t)ny * nx;
    int nck = (nz + SS_KCHUNK - 1) / SS_KCHUNK;
    wreal s1 = 0, s2 = 0, t1 = 0, t2 = 0;
    for (int ck = 0; ck < nck; ++ck) {
        for (int q = 0; q < 4; ++q) {
            int f = ss_face(q, i, nx, boundary_x);
            if (f < 0) continue;
            size_t fi = (size_t)ck * ny * nxf + (size_t)j * nxf + f;
            s1 += xm1[fi];
            s2 += xm2[fi];
        }
        for (int q = 0; q < 4; ++q) {
            int g = ss_face(q, j, ny, boundary_y);
            if (g < 0) continue;
            size_t fi = (size_t)ck * nyf * nx + (size_t)g * nx + i;
            s1 += ym1[fi];
            s2 += ym2[fi];
        }
        t1 += c1[(size_t)ck * st + c];
        t2 += c2[(size_t)ck * st + c];
    }
    if (finish) {
        pre[c] += m1[c];
    } else {
        for (int ck = 0; ck + 1 < nck; ++ck) {
            size_t f = (size_t)(ck + 1) * SS_KCHUNK * st + c;
            wreal e = edge[(size_t)ck * st + c];
            ph0_a[f] += e;
            ph_a[f] -= e;
        }
    }
    m1[c] += (wreal)0.5 * s1 + t1;
    m2[c] += (wreal)0.5 * s2 + t2;
}

// -------------------------------------------------------------- finish

#define SF_STATE(s)                                                         \
    wreal* __restrict__ u##s, wreal* __restrict__ v##s,                    \
    wreal* __restrict__ w##s, wreal* __restrict__ thp##s,                  \
    wreal* __restrict__ php##s, wreal* __restrict__ mup##s

// The adjoint's copy of the base state before the update.
#define SF_TRAJ                                                             \
    const wreal* __restrict__ u, const wreal* __restrict__ v,              \
    const wreal* __restrict__ w, const wreal* __restrict__ thp,            \
    const wreal* __restrict__ php, const wreal* __restrict__ mup

#define SF_IN(s)                                                            \
    const wreal* __restrict__ u_pp##s, const wreal* __restrict__ v_pp##s,  \
    const wreal* __restrict__ w_pp##s, const wreal* __restrict__ th_pp##s, \
    const wreal* __restrict__ ph_pp##s, const wreal* __restrict__ mu_pp##s,\
    const wreal* __restrict__ muts##s, const wreal* __restrict__ h_diabatic##s

#define SF_ADJ                                                              \
    wreal* __restrict__ u_pp_a, wreal* __restrict__ v_pp_a,                \
    wreal* __restrict__ w_pp_a, wreal* __restrict__ th_pp_a,               \
    wreal* __restrict__ ph_pp_a, wreal* __restrict__ mu_pp_a,              \
    wreal* __restrict__ muts_a, wreal* __restrict__ h_diabatic_a

#define SF_PASSIVE                                                          \
    const wreal* __restrict__ mub2d, const wreal* __restrict__ c1h,        \
    const wreal* __restrict__ c2h, const wreal* __restrict__ c1f,          \
    const wreal* __restrict__ c2f, const wreal* __restrict__ msfu,         \
    const wreal* __restrict__ msfv, const wreal* __restrict__ msft

static __device__ __forceinline__ void sf_face_mass(
    const wreal* mup, const wreal* muts, const wreal* mub2d, size_t ca,
    size_t cb, wreal* msf, wreal* mnf)
{
    *msf = (wreal)0.5 * (((mup[ca] + mup[cb]) + mub2d[ca]) + mub2d[cb]);
    *mnf = (wreal)0.5 * (muts[ca] + muts[cb]);
}

extern "C" __global__
void small_step_finish_uv_tl(SF_STATE(), SF_STATE(_d), SF_IN(), SF_IN(_d),
                             SF_PASSIVE, int nl, SS_GEOM)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;
    if (j < ny) {
        size_t ca = (size_t)j * nx + ss_cell_a(i, nx, boundary_x);
        size_t cb = (size_t)j * nx + ss_cell_b(i, nx, boundary_x);
        wreal msf, mnf;
        sf_face_mass(mup, muts, mub2d, ca, cb, &msf, &mnf);
        wreal msf_d = (wreal)0.5 * (mup_d[ca] + mup_d[cb]);
        wreal mnf_d = (wreal)0.5 * (muts_d[ca] + muts_d[cb]);
        wreal cs = c1h[k] * msf + c2h[k];
        wreal cn = c1h[k] * mnf + c2h[k];
        size_t ix = SI3(k, j, i, ny, nxf);
        wreal pp = u_pp[ix], pp_d = u_pp_d[ix];
        if (has_msf) {
            pp = pp * msfu[(size_t)j * nxf + i];
            pp_d = pp_d * msfu[(size_t)j * nxf + i];
        }
        wreal un = (cs * u[ix] + pp) / cn;
        wreal ud = (c1h[k] * msf_d * u[ix] + cs * u_d[ix] + pp_d
                    - un * c1h[k] * mnf_d) / cn;
        if (WRFPLUS_TL) u_d[ix] = ud;
        if (nl) u[ix] = un;
    }
    if (i < nx) {
        size_t ca = (size_t)ss_cell_a(j, ny, boundary_y) * nx + i;
        size_t cb = (size_t)ss_cell_b(j, ny, boundary_y) * nx + i;
        wreal msf, mnf;
        sf_face_mass(mup, muts, mub2d, ca, cb, &msf, &mnf);
        wreal msf_d = (wreal)0.5 * (mup_d[ca] + mup_d[cb]);
        wreal mnf_d = (wreal)0.5 * (muts_d[ca] + muts_d[cb]);
        wreal cs = c1h[k] * msf + c2h[k];
        wreal cn = c1h[k] * mnf + c2h[k];
        size_t ix = SI3(k, j, i, nyf, nx);
        wreal pp = v_pp[ix], pp_d = v_pp_d[ix];
        if (has_msf) {
            pp = pp * msfv[(size_t)j * nx + i];
            pp_d = pp_d * msfv[(size_t)j * nx + i];
        }
        wreal vn = (cs * v[ix] + pp) / cn;
        wreal vd = (c1h[k] * msf_d * v[ix] + cs * v_d[ix] + pp_d
                    - vn * c1h[k] * mnf_d) / cn;
        if (WRFPLUS_TL) v_d[ix] = vd;
        if (nl) v[ix] = vn;
    }
}

extern "C" __global__
void small_step_finish_column_tl(SF_STATE(), SF_STATE(_d), SF_IN(), SF_IN(_d),
                                 SF_PASSIVE, int nl, SS_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    size_t st = (size_t)ny * nx;
    wreal mus = mub2d[c] + mup[c];
    wreal mun = muts[c];
    wreal mus_d = mup_d[c], mun_d = muts_d[c];
    for (int k = 0; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        wreal pp = w_pp[f], pp_d = w_pp_d[f];
        if (has_msf) { pp = pp * msft[c]; pp_d = pp_d * msft[c]; }
        wreal cs = c1f[k] * mus + c2f[k];
        wreal cn = c1f[k] * mun + c2f[k];
        wreal wn = (cs * w[f] + pp) / cn;
        if (WRFPLUS_TL)
            w_d[f] = (c1f[k] * mus_d * w[f] + cs * w_d[f] + pp_d - wn * c1f[k] * mun_d) / cn;
        if (WRFPLUS_TL) php_d[f] = php_d[f] + ph_pp_d[f];
        if (nl) { w[f] = wn; php[f] = php[f] + ph_pp[f]; }
    }
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        wreal cs = c1h[k] * mus + c2h[k];
        wreal cs_d = c1h[k] * mus_d;
        wreal theta = thp[h];
        wreal numerator = th_pp[h];
        wreal num_d = th_pp_d[h];
        if (remove_hdiab) {
            wreal removal = (hdiab_dt * cs) * h_diabatic[h];
            numerator = numerator - removal;
            num_d -= hdiab_dt * (cs_d * h_diabatic[h] + cs * h_diabatic_d[h]);
        }
        numerator = numerator + theta * cs;
        num_d += thp_d[h] * cs + theta * cs_d;
        wreal cn = c1h[k] * mun + c2h[k];
        wreal tn = numerator / cn;
        if (WRFPLUS_TL) thp_d[h] = (num_d - tn * c1h[k] * mun_d) / cn;
        if (nl) thp[h] = tn;
    }
    if (WRFPLUS_TL) mup_d[c] = mup_d[c] + mu_pp_d[c];
    if (nl) mup[c] = mup[c] + mu_pp[c];
}

// Chunked column adjoint of small_step_finish_column (``w``, ``thp``,
// ``mup`` hold the pre-update values).  The mup/muts sums go to c1/c2;
// mu_pp's adjoint (mup's output adjoint) is added by the gather.
extern "C" __global__
void small_step_finish_column_ad(SF_STATE(_a), SF_TRAJ, SF_IN(), SF_ADJ,
                                 SS_CHUNK_SCRATCH, SF_PASSIVE, SS_GEOM)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    int nck = (nz + SS_KCHUNK - 1) / SS_KCHUNK;
    if (tid >= nck * ny * nx) return;
    int ck = tid / (ny * nx);
    int c = tid - ck * ny * nx;
    int k0 = ck * SS_KCHUNK;
    int k1 = min(nz, k0 + SS_KCHUNK);
    int kw = k1 == nz ? nz + 1 : k1;
    wreal mus = mub2d[c] + mup[c];
    wreal mun = muts[c];
    wreal a_mus = 0, a_mun = 0;
    for (int k = k0; k < kw; ++k) {
        size_t f = (size_t)k * st + c;
        wreal pp = w_pp[f];
        if (has_msf) pp = pp * msft[c];
        wreal cs = c1f[k] * mus + c2f[k];
        wreal cn = c1f[k] * mun + c2f[k];
        wreal wn = (cs * w[f] + pp) / cn;
        wreal an = w_a[f] / cn;
        a_mun -= an * wn * c1f[k];
        a_mus += an * w[f] * c1f[k];
        w_a[f] = an * cs;
        w_pp_a[f] += has_msf ? an * msft[c] : an;
        ph_pp_a[f] += php_a[f];
    }
    for (int k = k0; k < k1; ++k) {
        size_t h = (size_t)k * st + c;
        wreal cs = c1h[k] * mus + c2h[k];
        wreal cn = c1h[k] * mun + c2h[k];
        wreal numerator = th_pp[h];
        if (remove_hdiab) numerator = numerator - (hdiab_dt * cs) * h_diabatic[h];
        numerator = numerator + thp[h] * cs;
        wreal tn = numerator / cn;
        wreal an = thp_a[h] / cn;
        a_mun -= an * tn * c1h[k];
        th_pp_a[h] += an;
        wreal a_cs = an * thp[h];
        thp_a[h] = an * cs;
        if (remove_hdiab) {
            a_cs -= an * hdiab_dt * h_diabatic[h];
            h_diabatic_a[h] -= an * hdiab_dt * cs;
        }
        a_mus += a_cs * c1h[k];
    }
    c1[(size_t)ck * st + c] = a_mus;     // -> mup
    c2[(size_t)ck * st + c] = a_mun;     // -> muts
    (void)edge;
}

// Chunked face pass of small_step_finish_uv's adjoint (u, v hold the
// inputs): the msf (xm1/ym1) and mnf (xm2/ym2) chunk sums go to the gather.
extern "C" __global__
void small_step_finish_uv_ad(SF_STATE(_a), SF_TRAJ, SF_IN(), SF_ADJ, SS_SCRATCH,
                             SF_PASSIVE, SS_GEOM)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    int nck = (nz + SS_KCHUNK - 1) / SS_KCHUNK;
    if (tid >= nck * nyf * nxf) return;
    int ck = tid / (nyf * nxf);
    int r = tid - ck * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;
    int k0 = ck * SS_KCHUNK;
    int k1 = min(nz, k0 + SS_KCHUNK);
    if (j < ny) {
        size_t ca = (size_t)j * nx + ss_cell_a(i, nx, boundary_x);
        size_t cb = (size_t)j * nx + ss_cell_b(i, nx, boundary_x);
        wreal msf, mnf;
        sf_face_mass(mup, muts, mub2d, ca, cb, &msf, &mnf);
        wreal m = has_msf ? msfu[(size_t)j * nxf + i] : (wreal)1.0;
        wreal ams = 0, amn = 0;
        for (int k = k0; k < k1; ++k) {
            size_t ix = SI3(k, j, i, ny, nxf);
            wreal cs = c1h[k] * msf + c2h[k];
            wreal cn = c1h[k] * mnf + c2h[k];
            wreal pp = has_msf ? u_pp[ix] * m : u_pp[ix];
            wreal un = (cs * u[ix] + pp) / cn;
            wreal an = u_a[ix] / cn;
            amn -= an * un * c1h[k];
            ams += an * u[ix] * c1h[k];
            u_a[ix] = an * cs;
            u_pp_a[ix] += an * m;
        }
        size_t o = (size_t)ck * ny * nxf + (size_t)j * nxf + i;
        xm1[o] = ams;
        xm2[o] = amn;
    }
    if (i < nx) {
        size_t ca = (size_t)ss_cell_a(j, ny, boundary_y) * nx + i;
        size_t cb = (size_t)ss_cell_b(j, ny, boundary_y) * nx + i;
        wreal msf, mnf;
        sf_face_mass(mup, muts, mub2d, ca, cb, &msf, &mnf);
        wreal m = has_msf ? msfv[(size_t)j * nx + i] : (wreal)1.0;
        wreal ams = 0, amn = 0;
        for (int k = k0; k < k1; ++k) {
            size_t ix = SI3(k, j, i, nyf, nx);
            wreal cs = c1h[k] * msf + c2h[k];
            wreal cn = c1h[k] * mnf + c2h[k];
            wreal pp = has_msf ? v_pp[ix] * m : v_pp[ix];
            wreal vn = (cs * v[ix] + pp) / cn;
            wreal an = v_a[ix] / cn;
            amn -= an * vn * c1h[k];
            ams += an * v[ix] * c1h[k];
            v_a[ix] = an * cs;
            v_pp_a[ix] += an * m;
        }
        size_t o = (size_t)ck * nyf * nx + (size_t)j * nx + i;
        ym1[o] = ams;
        ym2[o] = amn;
    }
}
