// The smoke tracer's sources and sinks glue (woof/core/smoke_sources.py).
//
// Composed after glibc_flt32.cuh and smoke_plumerise.cu (the unit
// woof.core.smoke_sources.module_source() assembles): the plume entry below
// reuses the WRF-Chem plume model's device functions word for word, and the
// ambient Exner ratio is glibc's powf (gfk_pow).  Compiled directly by NVRTC
// with -fmad=false --ftz=false, every operation an explicit IEEE one, so the
// NumPy twins in woof/core/smoke_sources.py reproduce every output word
// (tests/test_smoke_sources.py).  The physics itself -- the injection
// profiles, Zhang's deposition velocity, GOCART settling, the aerodynamic
// resistance and the surface tendency -- is the merged woof.smoke library's
// own kernels, launched unchanged; these kernels only form their inputs and
// apply their outputs to the tracer.
//
// Every kernel is one thread per column (the plume entry one block per
// column), sums in increasing level order, and uses no atomics.

// --------------------------------------------------------------------------
// The ambient column, as WRF-Chem's chem_prep and plumerise_driver form it
// (chem/module_chem_utilities.F:79-129, chem/module_plumerise1.F:243-254):
//   z_at_w = (phb + ph) / g                  w levels, absolute
//   dz8w   = z_at_w(k+1) - z_at_w(k)
//   z      = 0.5 * (z_at_w(k) + z_at_w(k+1))
//   zmid   = z - z_at_w(kts)                 (zt)
//   z_lev  = z_at_w(k) - z_at_w(kts)         (zw, nz + 1 levels here)
//   ratio  = (p_phy / p1000mb) ** rcp        (glibc powf)
//   t_phy  = theta * ratio
//   pi_in  = cp * ratio
//   theta_in = t_phy / pi_in * cp
//   rho    = 1. / alt * (1. + qv)
//   u_phy  = 0.5 * (u(i) + u(i+1)),  v_phy likewise
// theta is woof's full potential temperature thb + thp (WRF's t + t0) and
// p_phy is woof's full pressure.  Any output pointer may be null: only the
// fields a caller asks for are written.  zref is zmid at the lowest level.
extern "C" __global__ void smoke_source_env(
    const float *__restrict__ u, const float *__restrict__ v,
    const float *__restrict__ thb, int thb_3d,
    const float *__restrict__ thp, const float *__restrict__ p,
    const float *__restrict__ qv, const float *__restrict__ phb, int phb_3d,
    const float *__restrict__ php, const float *__restrict__ alt,
    float g, float rcp, float cp, float p1000,
    float *__restrict__ u_out, float *__restrict__ v_out,
    float *__restrict__ theta_out, float *__restrict__ pi_out,
    float *__restrict__ qv_out, float *__restrict__ zt_out,
    float *__restrict__ zw_out, float *__restrict__ dz_out,
    float *__restrict__ t_out, float *__restrict__ rho_out,
    float *__restrict__ zref_out, int nz, int ny, int nx)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int ncol = ny * nx;
    if (c >= ncol) return;
    const int j = c / nx, i = c - j * nx;
    const size_t nc = (size_t)ncol;
    const float zs = __fdiv_rn(__fadd_rn(phb_3d ? phb[c] : phb[0], php[c]), g);
    float zlo = zs;
    if (zw_out) zw_out[c] = __fsub_rn(zs, zs);
    for (int k = 0; k < nz; ++k) {
        const size_t at = (size_t)k * nc + c;
        const size_t up = (size_t)(k + 1) * nc + c;
        const float zhi = __fdiv_rn(
            __fadd_rn(phb_3d ? phb[up] : phb[k + 1], php[up]), g);
        if (zw_out) zw_out[up] = __fsub_rn(zhi, zs);
        if (dz_out) dz_out[at] = __fsub_rn(zhi, zlo);
        const float zmid = __fsub_rn(__fmul_rn(0.5f, __fadd_rn(zlo, zhi)), zs);
        if (zt_out) zt_out[at] = zmid;
        if (zref_out && k == 0) zref_out[c] = zmid;
        const float q = qv ? qv[at] : 0.0f;
        if (qv_out) qv_out[at] = q;
        if (theta_out || pi_out || t_out) {
            const float theta = __fadd_rn(thb_3d ? thb[at] : thb[k], thp[at]);
            const float ratio = gfk_pow(__fdiv_rn(p[at], p1000), rcp);
            const float t_phy = __fmul_rn(theta, ratio);
            const float pi_in = __fmul_rn(cp, ratio);
            if (t_out) t_out[at] = t_phy;
            if (pi_out) pi_out[at] = pi_in;
            if (theta_out)
                theta_out[at] = __fmul_rn(__fdiv_rn(t_phy, pi_in), cp);
        }
        if (rho_out)
            rho_out[at] = __fmul_rn(__fdiv_rn(1.0f, alt[at]),
                                    __fadd_rn(1.0f, q));
        if (u_out) {
            const size_t row = ((size_t)k * ny + j) * (size_t)(nx + 1);
            u_out[at] = __fmul_rn(0.5f, __fadd_rn(u[row + i], u[row + i + 1]));
        }
        if (v_out) {
            const size_t row = (size_t)k * (size_t)(ny + 1) * nx;
            v_out[at] = __fmul_rn(0.5f, __fadd_rn(v[row + (size_t)j * nx + i],
                                                 v[row + (size_t)(j + 1) * nx + i]));
        }
        zlo = zhi;
    }
}

// --------------------------------------------------------------------------
// The plume-rise injection over every column of a domain, driven by a fire's
// own heat flux and burning area.  The body is smoke_plumerise_columns'
// (smoke_plumerise.cu, plumerise lines 10-205 with set_flam_vert's k1/k2
// search) with two differences, both in its inputs:
//   * the heat-flux table is per column: the column's fire is WRF's
//     single-heat-flux biome (aggregated biome 4, one plume, injected from
//     level 3 to the level above the plume top), its table entry the fire's
//     heat flux in kW m-2 (W m-2 / 1000.);
//   * a column runs when it burns -- heat flux > 0, and the driver's own
//     tests (module_plumerise1.F:213-227) on the flaming fraction, the fire
//     size and a non-zero emission -- with eburn_in = 1, so the profile is
//     the emission's shape and any flux the step carries scales it.  Every
//     other column returns at once, so the launch is one block per column
//     with no host compaction (and no device-to-host read).
// heat_flux/area are the fire's planes (W m-2, m2), or null: then a column
// whose emission is positive takes the prescribed scalars.
extern "C" __global__ void smoke_plume_fire_columns(
    int m1, int ncol, const float *u, const float *v, const float *theta,
    const float *pi, const float *qv, const float *zt, const float *zm,
    const float *__restrict__ heat_flux, const float *__restrict__ area,
    const float *__restrict__ emis, float heat_flux_w, float area_m2,
    float flaming, float *__restrict__ eburn_out,
    float *__restrict__ ztop_out, int *__restrict__ k1_out,
    int *__restrict__ k2_out, int *__restrict__ status)
{
    extern __shared__ float sp_shared[];
    __shared__ SpScal S;
    __shared__ float table[8];
    const int col = blockIdx.x;
    const size_t nc = (size_t)ncol;
    for (int k = threadIdx.x; k < m1; k += blockDim.x)
        eburn_out[(size_t)k * nc + col] = (k == 0) ? 1.0f : 0.0f;
    float h = 0.0f, a = 0.0f;
    if (heat_flux) {
        h = heat_flux[col];
        a = area[col];
    } else if (emis[col] > 0.0f) {
        h = heat_flux_w;
        a = area_m2;
    }
    if (threadIdx.x == 0) {
        ztop_out[col] = 0.0f;
        k1_out[col] = 0;
        k2_out[col] = 0;
        status[col] = 0;
    }
    // module_plumerise1.F:213-227 with eburn_in = 1: the flaming-fraction
    // sum and the fire-size sum, each against 1.e-6.
    const float sum_fct = ((0.0f + 0.0f) + 0.0f) + flaming;
    const float sum_size = ((0.0f + 0.0f) + 0.0f) + a;
    if (!(h > 0.0f) || sum_fct < 1.e-6f || sum_size < 1.e-6f
        || emis[col] == 0.0f) return;
    const float hk = __fdiv_rn(h, 1000.0f);
    if (threadIdx.x < 8) table[threadIdx.x] = hk;

    SpCol c;
    c.s = sp_shared;
    c.S = &S;
    c.tx = threadIdx.x;
    c.nt = blockDim.x;
    c.u = u; c.v = v; c.theta = theta; c.pi = pi; c.qv = qv;
    c.zt = zt; c.zm = zm;
    c.m1 = m1;
    c.ncol = ncol;
    c.col = col;
    c.heat_flux = table;

    // zero_plumegen_coms: the first-call state.
    for (int i = c.tx; i < SP_NARR * SP_LEV; i += c.nt) sp_shared[i] = 0.0f;
    if (c.tx == 0) {
        S.wbar = S.dqsdz = S.viscosity = S.tstpf = S.ztop = 0.0f;
        S.area = S.rsurf = S.alpha = S.fmoist = S.dt = S.time = 0.0f;
        S.tdur = S.heat_flux_w = S.wmax = S.ztopmax = 0.0f;
        S.mintime = S.mdur = S.maxtime = S.nm1 = S.kkmax = 0;
        S.done = S.status = S.kmt = 0;
    }
    SP_SYNC();
    sp_get_env_condition(c);
    if (S.status) {
        if (c.tx == 0) status[col] = S.status;
        return;
    }
    const float ein = 1.0f;
    const float mean_fct[4] = {0.0f, 0.0f, 0.0f, flaming};
    const float firesize[4] = {0.0f, 0.0f, 0.0f, a};
    float ztopmax[3] = {0.0f, 0.0f, 0.0f};
    for (int iveg = 1; iveg <= 4; ++iveg) {
        const float fct = mean_fct[iveg - 1];
        if (fct < 1.e-6f) continue;
        const float burnt_area = firesize[iveg - 1];
        for (int imm = 1; imm <= 2; ++imm) {
            if (c.tx == 0) sp_get_fire_properties(c, imm, iveg, burnt_area);
            SP_SYNC();
            if (iveg == 4 && imm == 2) {
                ztopmax[2] = ztopmax[1];
                ztopmax[1] = sp_zzcon(c, 1);
                continue;
            }
            sp_makeplume(c);
            ztopmax[imm] = S.ztopmax;
            SP_SYNC();
        }
        if (c.tx == 0) {
            int klim[3];
            for (int imm = 1; imm <= 2; ++imm) {
                int k;
                for (k = 1; k <= SP_NKP - 1; ++k)
                    if (sp_zzcon(c, k) > ztopmax[imm]) break;
                klim[imm] = k;
            }
            const int k1 = 3 > klim[1] ? 3 : klim[1];
            int k2 = 3 > klim[2] ? 3 : klim[2];
            if (k2 < k1) k2 = k1;
            const float dz_flam = sp_zzcon(c, k2 + 1) - sp_zzcon(c, k1);
            const float dzi = 1.0f / dz_flam;
            const float add = (fct * ein) * dzi;
            for (int k = k1; k <= k2 && k <= m1; ++k)
                eburn_out[(size_t)(k - 1) * nc + col] =
                    eburn_out[(size_t)(k - 1) * nc + col] + add;
            if (k2 >= m1) status[col] = 3;
            ztop_out[col] = ztopmax[2];
            k1_out[col] = k1;
            k2_out[col] = k2;
        }
    }
}

// The injection top: the top (m above ground) of the highest layer the
// column's emission enters.
__device__ __forceinline__ float ss_injection_top(
    const float *frac, const float *zw, size_t col, size_t nc, int nz)
{
    int top = 0;
    for (int k = 0; k < nz; ++k)
        if (frac[(size_t)k * nc + col] != 0.0f) top = k;
    return zw[(size_t)(top + 1) * nc + col];
}

// The plume profile as fractions of the column's emission, WRF's own
// conversion and scale_fire_emiss normalization (module_plumerise1.F):
//   ebu(kts) = eburn_out(1)
//   ebu(k)   = eburn_out(k) * (z_at_w(k+1) - z_at_w(k)),  k = kts+1..kte
//   sum      = ebu(kts) + ... + ebu(kte);  ratio = ebu(kts) / sum
//   frac(k)  = ebu(k) * ratio
// A column the plume model refused (status != 0, where WRF stops) keeps the
// whole emission in the lowest layer and reports a NaN injection top, which
// the health census names at the next history write.
extern "C" __global__ void smoke_plume_fraction(
    const float *__restrict__ eburn_out, const float *__restrict__ dz,
    const float *__restrict__ zw, const int *__restrict__ status,
    float *__restrict__ frac, float *__restrict__ inj_top, int nz, int ncol)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const size_t nc = (size_t)ncol;
    if (status[c] != 0) {
        for (int k = 0; k < nz; ++k)
            frac[(size_t)k * nc + c] = (k == 0) ? 1.0f : 0.0f;
        inj_top[c] = __int_as_float(0x7fc00000);
        return;
    }
    float sum = eburn_out[c];
    for (int k = 1; k < nz; ++k) {
        const size_t at = (size_t)k * nc + c;
        sum = __fadd_rn(sum, __fmul_rn(eburn_out[at], dz[at]));
    }
    const float ratio = sum > 0.0f ? __fdiv_rn(eburn_out[c], sum) : 0.0f;
    frac[c] = __fmul_rn(eburn_out[c], ratio);
    for (int k = 1; k < nz; ++k) {
        const size_t at = (size_t)k * nc + c;
        frac[at] = __fmul_rn(__fmul_rn(eburn_out[at], dz[at]), ratio);
    }
    inj_top[c] = ss_injection_top(frac, zw, c, nc, nz);
}

// The boundary-layer profile as fractions: the library's
// pbl_uniform_profile, run with a unit flux and a unit density, gives
// S(k) = share(k) / dz(k); the fraction is S(k) * dz(k) on the same z_w.
extern "C" __global__ void smoke_pbl_fraction(
    const float *__restrict__ source, const float *__restrict__ zw,
    float *__restrict__ frac, float *__restrict__ inj_top, int nz, int ncol)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const size_t nc = (size_t)ncol;
    for (int k = 0; k < nz; ++k) {
        const size_t at = (size_t)k * nc + c;
        frac[at] = __fmul_rn(source[at],
                             __fsub_rn(zw[at + nc], zw[at]));
    }
    inj_top[c] = ss_injection_top(frac, zw, c, nc, nz);
}

// The emission through a fraction profile as a coupled tendency:
//   tend(k) += ((emis * frac(k)) * ug_g) / (-dnw(k))
// smoke_surface_source's statement with the fraction folded in: a column
// whose profile is (1, 0, ..., 0) adds the same words to the lowest layer.
// A column with no profile yet (all zero: a state whose first plume call is
// still ahead) takes the surface profile, so no emission is ever dropped.
extern "C" __global__ void smoke_profile_source(
    float *__restrict__ tend, const float *__restrict__ emis,
    const float *__restrict__ frac, const float *__restrict__ dnw,
    float ug_g, int nz, int ncol)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const size_t nc = (size_t)ncol;
    const float e = emis[c];
    bool any = false;
    for (int k = 0; k < nz; ++k)
        if (frac[(size_t)k * nc + c] != 0.0f) any = true;
    if (!any) {
        const float depth = -dnw[0];
        tend[c] = __fadd_rn(tend[c], __fdiv_rn(__fmul_rn(e, ug_g), depth));
        return;
    }
    for (int k = 0; k < nz; ++k) {
        const size_t at = (size_t)k * nc + c;
        const float f = frac[at];
        if (k > 0 && f == 0.0f) continue;
        const float depth = -dnw[k];
        const float source = __fdiv_rn(__fmul_rn(__fmul_rn(e, f), ug_g),
                                       depth);
        tend[at] = __fadd_rn(tend[at], source);
    }
}

// --------------------------------------------------------------------------
// Dry deposition inputs, per column, for zhang2001_aer_drydepvel_1 as
// aer_drydep_driver calls it (module_aer_drydep.F:319-354) and for the
// aerodynamic resistance: the lowest level's temperature and density, the
// lowest mass level's height, the Obukhov length 1/rmol (infinite, neutral,
// without one), the USGS land-use class (MODIS mapped by WRF's own
// luse2usgs, module_dep_simple.F:1965-1967), the season and a dry surface.
// The particle is one monodisperse bin (GOCART's): number diameter 2*r and
// ln(sigma) = 0.  airkinvisc and freepath are zeroed, as the driver passes
// them, so the routine forms both.
__constant__ int SS_MODIS_TO_USGS[21] = {
    14, 13, 12, 11, 15, 8, 9, 10, 10, 7, 17, 4, 1, 5, 24, 19, 16, 21, 22, 23,
    16};

extern "C" __global__ void smoke_drydep_prepare(
    const float *__restrict__ t_phy, const float *__restrict__ rho,
    const float *__restrict__ rmol, const int *__restrict__ ivgtyp,
    int modis, int iseason, float diameter, float density,
    float *__restrict__ dg_num, float *__restrict__ lnsig,
    float *__restrict__ dens, float *__restrict__ airtemp,
    float *__restrict__ airdens, float *__restrict__ obukhov,
    int *__restrict__ season, int *__restrict__ surftype,
    int *__restrict__ wetsurf, float *__restrict__ kinvisc,
    float *__restrict__ freepath, int ncol)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    dg_num[c] = diameter;
    lnsig[c] = 0.0f;
    dens[c] = density;
    airtemp[c] = t_phy[c];
    airdens[c] = rho[c];
    obukhov[c] = rmol ? __fdiv_rn(1.0f, rmol[c])
                      : __int_as_float(0x7f800000);
    int lu = ivgtyp[c];
    if (modis) lu = (lu >= 1 && lu <= 21) ? SS_MODIS_TO_USGS[lu - 1] : 0;
    surftype[c] = lu;
    season[c] = iseason;
    wetsurf[c] = 0;
    // airkinvisc = 0. and freepath = 0.: the routine computes both.
    kinvisc[c] = 0.0f;
    freepath[c] = 0.0f;
}

// dry_dep_driver's cap on aerosol deposition velocities
// (dry_dep_driver.F:624-631): ddvel = MIN(0.50, ddvel), gfortran's
// a < b ? a : b with a = 0.50.
extern "C" __global__ void smoke_drydep_cap(
    const float *__restrict__ depvel, float *__restrict__ capped, int ncol)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const float v = depvel[c];
    capped[c] = 0.50f < v ? 0.50f : v;
}

// The lowest layer's coupled dry air mass, kg m-2, as smoke_column_load
// forms it.
__device__ __forceinline__ float ss_layer_mass(
    const float *c1h, const float *c2h, const float *dnw, float mu, int k,
    float g)
{
    const float chm = __fadd_rn(__fmul_rn(c1h[k], mu), c2h[k]);
    return __fmul_rn(chm, __fdiv_rn(-dnw[k], g));
}

// One forward step of the library's lowest-layer tendency, held
// non-negative, and the mass it removed booked as deposition (ug m-2):
//   q0' = max(q0 + dt * tendency, 0)
//   dep += M0 * (q0 - q0')
extern "C" __global__ void smoke_drydep_apply(
    float *__restrict__ smoke, const float *__restrict__ tendency,
    const float *__restrict__ c1h, const float *__restrict__ c2h,
    const float *__restrict__ dnw, const float *__restrict__ mub,
    const float *__restrict__ mup, float *__restrict__ deposited,
    float dt, float g, int ncol)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const float q0 = smoke[c];
    float qn = __fadd_rn(q0, __fmul_rn(dt, tendency[c]));
    if (qn < 0.0f) qn = 0.0f;
    const float mu = __fadd_rn(mub[c], mup[c]);
    const float layer = ss_layer_mass(c1h, c2h, dnw, mu, 0, g);
    deposited[c] = __fadd_rn(deposited[c],
                             __fmul_rn(layer, __fsub_rn(q0, qn)));
    smoke[c] = qn;
}

// GOCART settling's REAL*8 result back into the real(4) tracer, and the
// column mass it removed (ug m-2) booked as deposition -- dry_dep_driver's
// own bookkeeping, old minus new column totals, never negative
// (dry_dep_driver.F:784-809):
//   loss = sum_k M_k * (q_k - real(tc_k))   (REAL*8, k upward)
//   dep += real(max(loss, 0)),  graset += the same
extern "C" __global__ void smoke_settling_apply(
    float *__restrict__ smoke, const double *__restrict__ tc,
    const float *__restrict__ c1h, const float *__restrict__ c2h,
    const float *__restrict__ dnw, const float *__restrict__ mub,
    const float *__restrict__ mup, float *__restrict__ deposited,
    float *__restrict__ graset, float g, int nz, int ncol)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const size_t nc = (size_t)ncol;
    const float mu = __fadd_rn(mub[c], mup[c]);
    double loss = 0.0;
    for (int k = 0; k < nz; ++k) {
        const size_t at = (size_t)k * nc + c;
        const float q = smoke[at];
        const float qn = __double2float_rn(tc[at]);
        const double layer = (double)ss_layer_mass(c1h, c2h, dnw, mu, k, g);
        loss = __dadd_rn(loss, __dmul_rn(layer,
                                         __dsub_rn((double)q, (double)qn)));
        smoke[at] = qn;
    }
    const float booked = __double2float_rn(loss > 0.0 ? loss : 0.0);
    deposited[c] = __fadd_rn(deposited[c], booked);
    graset[c] = __fadd_rn(graset[c], booked);
}
