// Smoke particle removal from WRF v4.7.1, plus two woof-own pieces.
//
//   smoke_deposition_zhang2001        chem/module_aer_drydep.F:1650-1941,
//                                     zhang2001_aer_drydepvel_1: the Zhang et
//                                     al. (2001) size-resolved particle dry
//                                     deposition velocity, real(4), one
//                                     thread per cell.
//   smoke_deposition_gocart_settling  chem/module_gocart_settling.F:206-367,
//                                     settling, REAL*8, fed one column at a
//                                     time as gocart_settling_driver feeds it
//                                     for dust (idust = 1, :89-122), one
//                                     thread per column.
//   smoke_deposition_resist_a         woof-own (no WRF oracle): aerodynamic
//                                     resistance from u*, z0, z_ref and an
//                                     Obukhov length, real(4).
//   smoke_deposition_surface_tendency woof-own: the lowest layer's
//                                     dry-deposition tendency and flux.
//
// Every WRF statement follows WRF's own, in WRF's order of evaluation; the
// unit is compiled with -fmad=false so no product is fused into a sum, as
// gfortran fuses none at -O0.  real(4) EXP and real powers (including
// x**beta with the real beta = 2.0) are glibc's expf and powf (gfk_exp,
// gfk_pow); integer powers are products; SQRT is the correctly rounded
// square root.  The REAL*8 routine's EXP and ** are glibc's exp and pow
// (g64_exp::exp, g64_pow::pow).  gfortran folds a constant subexpression
// (4.0/9.0, 3.0*pi, log(5.0)) once, correctly rounded, in its own kind; those
// are spelled here as the same single rounding.  A real(4) literal in a
// REAL*8 expression enters as its float value promoted (1.458E-6, 110.4,
// 1.257, 0.4, 1.1, 1.1E-3), exactly as gfortran promotes it.

// ---------------------------------------------------------------------------
// zhang2001_aer_drydepvel_1
// ---------------------------------------------------------------------------

__device__ const float SDZ_GAMMAX[15] = {
    0.56f, 0.58f, 0.56f, 0.56f, 0.56f,
    0.54f, 0.54f, 0.54f, 0.54f, 0.54f,
    0.54f, 0.54f, 0.50f, 0.50f, 0.56f};
__device__ const float SDZ_ALPHA[15] = {
    1.0f, 0.6f, 1.1f, 0.8f, 0.8f,
    1.2f, 1.2f, 50.0f, 50.0f, 1.3f,
    2.0f, 50.0f, 100.0f, 100.0f, 1.5f};
// rad_collector(5,15): one row of five seasons per land-use class.
__device__ const float SDZ_RAD_COLLECTOR[15][5] = {
    {0.002f, 0.002f, 0.002f, 0.002f, 0.002f},
    {0.005f, 0.005f, 0.005f, 0.005f, 0.005f},
    {0.002f, 0.002f, 0.005f, 0.005f, 0.002f},
    {0.005f, 0.005f, 0.010f, 0.010f, 0.005f},
    {0.005f, 0.005f, 0.005f, 0.005f, 0.005f},
    {0.002f, 0.002f, 0.005f, 0.005f, 0.002f},
    {0.002f, 0.002f, 0.005f, 0.005f, 0.002f},
    {9.999f, 9.999f, 9.999f, 9.999f, 9.999f},
    {9.999f, 9.999f, 9.999f, 9.999f, 9.999f},
    {0.010f, 0.010f, 0.010f, 0.010f, 0.010f},
    {0.010f, 0.010f, 0.010f, 0.010f, 0.010f},
    {9.999f, 9.999f, 9.999f, 9.999f, 9.999f},
    {9.999f, 9.999f, 9.999f, 9.999f, 9.999f},
    {9.999f, 9.999f, 9.999f, 9.999f, 9.999f},
    {0.010f, 0.010f, 0.010f, 0.010f, 0.010f}};
__device__ const int SDZ_ZHANG_FROM_USGS[25] = {
    15, 7, 7, 7, 7, 10, 6, 6, 6, 6, 4, 3, 2, 1, 5,
    14, 11, 10, 8, 9, 10, 9, 8, 12, 8};

// gfortran's MAX/MIN of two reals: the second argument unless the first
// beats it, which is what max(a, min(b, x)) needs for ordinary values.
__device__ __forceinline__ float sdz_minf(float a, float b) { return b < a ? b : a; }
__device__ __forceinline__ float sdz_maxf(float a, float b) { return b > a ? b : a; }

// One moment of the routine's DO moment = 0, 3, 3 loop.
__device__ void sdz_zhang_moment(
    float moment, float dg_num, float lnsig, float dens_part, float airtemp,
    float ustar, float resist_a, int isurftype, int iseason, int iwetsurf,
    int zhang_optaa, float e0_brown, float e0_impact, float e0_intcept,
    float airkinvisc, float freepath, float viscosity, float dp_vm,
    float *depvel, float *vsettl_out, float *resist_out)
{
    const float gravit = 9.806f;
    const float boltzmann = 1.381e-23f;
    const float pi = 3.1415926536f;
    float dp_mom = dg_num * gfk_exp(moment * lnsig * lnsig);
    dp_mom = sdz_minf(100.0e-6f, dp_mom);

    float tmpa = 2.0f * freepath / dp_mom;
    const float cuncorrect = 1.f + tmpa * (1.257f + 0.4f * gfk_exp(-(1.10f / tmpa)));
    const float dispersion = gfk_exp(2.f * lnsig * lnsig);
    const float vsettl = (dens_part * (dp_mom * dp_mom) * gravit *
                          cuncorrect / (18.0f * viscosity)) * dispersion;

    const float diffus_part = boltzmann * airtemp * cuncorrect /
                              ((3.0f * pi) * viscosity * dp_mom);
    const float schmidt = airkinvisc / diffus_part;
    const float e_brown = gfk_pow(schmidt, -SDZ_GAMMAX[isurftype - 1]);

    float stokes, e_intcept;
    if (isurftype == 8 || isurftype == 9 || isurftype == 12 ||
        isurftype == 13 || isurftype == 14) {
        stokes = vsettl * ustar * ustar / (gravit * airkinvisc);
        e_intcept = 0.f;
    } else {
        const float rc = SDZ_RAD_COLLECTOR[isurftype - 1][iseason - 1];
        stokes = vsettl * ustar / (gravit * rc);
        if (zhang_optaa == 2) {
            tmpa = 0.5f * dp_mom / rc;
            e_intcept = tmpa;
        } else {
            tmpa = dp_mom / rc;
            e_intcept = 0.5f * (tmpa * tmpa);
        }
    }

    const float beta = 2.0f;
    const float e_impact = gfk_pow(stokes / (SDZ_ALPHA[isurftype - 1] + stokes), beta);

    float stickfrac;
    if ((iwetsurf > 0) || (isurftype == 13) || (isurftype == 14) ||
        (dp_vm < 5.0e-6f)) {
        stickfrac = 1.0f;
    } else {
        stickfrac = gfk_exp(-sqrtf(stokes));
    }

    tmpa = ustar * (e0_brown * e_brown + e0_intcept * e_intcept +
                    e0_impact * e_impact) * stickfrac;
    const float resist_sfc = 1.f / sdz_maxf(tmpa, 1.0e-30f);

    *depvel = vsettl + 1.f / (resist_a + resist_sfc + resist_a * resist_sfc * vsettl);
    *vsettl_out = vsettl;
    *resist_out = resist_sfc;
}

extern "C" __global__ void smoke_deposition_zhang2001(
    const float *__restrict__ dg_num_in, const float *__restrict__ lnsig_in,
    const float *__restrict__ dens_part_in, const float *__restrict__ airtemp_in,
    const float *__restrict__ airdens_in, const float *__restrict__ ustar_in,
    const float *__restrict__ resist_a_in, const int *__restrict__ iseason_in,
    const int *__restrict__ isurftype_in, const int *__restrict__ iwetsurf_in,
    float *__restrict__ airkinvisc_io, float *__restrict__ freepath_io,
    float *__restrict__ depvel_0, float *__restrict__ depvel_3,
    float *__restrict__ resist_s0, float *__restrict__ resist_s3,
    float *__restrict__ vsettl_0, float *__restrict__ vsettl_3,
    int zhang_optaa, int modesurftype, int ido_mom0, int ido_mom3, int n)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;

    // reality checks; log(5.0) is folded by gfortran, correctly rounded.
    const float log5 = 0x1.9c0420p+0f;
    const float airtemp = sdz_maxf(50.0f, sdz_minf(350.0f, airtemp_in[c]));
    const float airdens = sdz_maxf(0.01f, sdz_minf(5.0f, airdens_in[c]));
    const float dg_num = sdz_maxf(1.0e-9f, sdz_minf(1.0e-2f, dg_num_in[c]));
    const float dens_part = sdz_maxf(0.1e3f, sdz_minf(50.0e3f, dens_part_in[c]));
    const float lnsig = sdz_maxf(0.0f, sdz_minf(log5, lnsig_in[c]));
    const float resist_a = sdz_maxf(1.0e-2f, sdz_minf(1.0e6f, resist_a_in[c]));
    const float ustar = sdz_maxf(1.0e-6f, sdz_minf(1.0e2f, ustar_in[c]));

    int isurftype = 8;
    const int ist = isurftype_in[c];
    if (modesurftype <= 1) {
        if ((ist >= 1) && (ist <= 15)) isurftype = ist;
    } else {
        if ((ist >= 1) && (ist <= 25)) isurftype = SDZ_ZHANG_FROM_USGS[ist - 1];
    }
    int iseason = 1;
    if ((iseason_in[c] >= 1) && (iseason_in[c] <= 5)) iseason = iseason_in[c];

    float e0_brown, e0_impact, e0_intcept;
    if (zhang_optaa == 2) {
        e0_brown = 1.0f; e0_impact = 1.0f; e0_intcept = 4.0f;
    } else {
        e0_brown = 3.0f; e0_impact = 3.0f; e0_intcept = 3.0f;
    }

    float freepath = freepath_io[c];
    if (freepath <= 0.0f) freepath = 6.6328e-8f * (1.2255f / airdens);
    float airkinvisc = airkinvisc_io[c];
    float viscosity;
    if (airkinvisc <= 0.0f) {
        viscosity = 1.458e-6f * airtemp * sqrtf(airtemp) / (airtemp + 110.4f);
        airkinvisc = viscosity / airdens;
    } else {
        viscosity = airkinvisc * airdens;
    }
    const float dp_vm = dg_num * gfk_exp(1.5f * lnsig * lnsig);

    if (ido_mom0 <= 0) {
        depvel_0[c] = 0.0f; vsettl_0[c] = 0.0f; resist_s0[c] = 1.0e30f;
    } else {
        float d, v, r;
        sdz_zhang_moment(0.0f, dg_num, lnsig, dens_part, airtemp, ustar,
                         resist_a, isurftype, iseason, iwetsurf_in[c],
                         zhang_optaa, e0_brown, e0_impact, e0_intcept,
                         airkinvisc, freepath, viscosity, dp_vm, &d, &v, &r);
        depvel_0[c] = d; vsettl_0[c] = v; resist_s0[c] = r;
    }
    if (ido_mom3 <= 0) {
        depvel_3[c] = 0.0f; vsettl_3[c] = 0.0f; resist_s3[c] = 1.0e30f;
    } else {
        float d, v, r;
        sdz_zhang_moment(3.0f, dg_num, lnsig, dens_part, airtemp, ustar,
                         resist_a, isurftype, iseason, iwetsurf_in[c],
                         zhang_optaa, e0_brown, e0_impact, e0_intcept,
                         airkinvisc, freepath, viscosity, dp_vm, &d, &v, &r);
        depvel_3[c] = d; vsettl_3[c] = v; resist_s3[c] = r;
    }
    airkinvisc_io[c] = airkinvisc;
    freepath_io[c] = freepath;
}

// ---------------------------------------------------------------------------
// GOCART settling (REAL*8), dust path
// ---------------------------------------------------------------------------

// gfortran's INT of a REAL*8 on x86-64 (cvttsd2si): truncation, and the
// "integer indefinite" INT_MIN for anything outside the int32 range.
__device__ __forceinline__ int sdz_int_x86(double x)
{
    if (!(x > -2147483649.0 && x < 2147483648.0)) return (int)0x80000000;
    return (int)x;
}

// One thread per column.  Inputs are WRF's real(4) fields in model order
// (level 0 at the ground): chem (nbin, nz, ncol), t_phy, p_phy, dz8w,
// rho_phy (nz, ncol).  The driver's reversed p_mid/delz index l2 = lmx-l+1
// lands on model level l, so both are read at level l here.  Outputs are the
// routine's REAL*8 results: tc (nbin, nz, ncol), graset and grasetvel
// (nbin, ncol).
extern "C" __global__ void smoke_deposition_gocart_settling(
    const float *__restrict__ chem, const float *__restrict__ t_phy,
    const float *__restrict__ p_phy, const float *__restrict__ dz8w,
    const float *__restrict__ rho_phy, const double *__restrict__ reff_in,
    const double *__restrict__ den_in, double *__restrict__ tc,
    double *__restrict__ graset, double *__restrict__ grasetvel,
    float dt, float g0, float dyn_visc, int uoc, int nbin, int nz, int ncol)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const size_t plane = (size_t)nz * (size_t)ncol;

    // WHERE (tc < 0.0) tc = 1.0D-32
    for (int b = 0; b < nbin; ++b)
        for (int k = 0; k < nz; ++k) {
            const size_t at = (size_t)b * plane + (size_t)k * ncol + col;
            const double v = (double)chem[at];
            tc[at] = (v < 0.0) ? 1.0e-32 : v;
        }
    double dzmin = (double)dz8w[col];
    for (int k = 1; k < nz; ++k) {
        const double d = (double)dz8w[(size_t)k * ncol + col];
        if (d < dzmin) dzmin = d;
    }
    const double growth_fac = 1.0;
    const float c49 = __fdiv_rn(4.0f, 9.0f);
    const float c29 = __fdiv_rn(2.0f, 9.0f);

    for (int b = 0; b < nbin; ++b) {
        double den = den_in[b];
        if (uoc == 1 && b == 0) den = 2650.0;
        const double reff = reff_in[b];
        const double gr = growth_fac * reff;
        const double vsettl = (double)(c49 * g0) * den * (gr * gr) / (double)dyn_visc;
        const int ntdt = (int)dt;
        const double dtmax = dzmin / vsettl;
        int ndt = sdz_int_x86((double)ntdt / dtmax);
        if (ndt < 1) ndt = 1;
        if (ndt > 12) ndt = 12;
        const double dt_settl = (double)((float)ntdt / (float)ndt);
        const double rwet = reff;
        const double rho = den;

        double gset = 0.0, gvel = 0.0;
        for (int it = 1; it <= ndt; ++it) {
            double transfer = 0.0;
            for (int l = nz; l >= 1; --l) {
                const size_t lv = (size_t)(l - 1) * ncol + col;
                const double tmp = (double)t_phy[lv];
                const double p_mid = (double)(0.01f * p_phy[lv]);
                const double delz = (double)dz8w[lv];
                const double c_stokes = (double)1.458E-6f * g64_pow::pow(tmp, 1.5) /
                                        (tmp + (double)110.4f);
                const double free_path = (double)1.1E-3f / p_mid / sqrt(tmp);
                const double c_cun = 1.0 + free_path / rwet *
                    ((double)1.257f + (double)0.4f *
                     g64_exp::exp(-((double)1.1f * rwet / free_path)));
                const double viscosity = c_stokes / c_cun;
                const double vd_cor = (double)(c29 * g0) * rho * (rwet * rwet) / viscosity;
                const size_t at = (size_t)b * plane + lv;
                const double temp_tc = tc[at];
                const double vd_wk1 = dt_settl * vd_cor / delz;
                tc[at] = tc[at] * (1.0 - vd_wk1) + transfer;
                if (l == 1) {
                    gset = gset + vd_cor * temp_tc / (double)ndt;
                    gvel = vd_cor;
                } else {
                    const size_t lb = (size_t)(l - 2) * ncol + col;
                    transfer = (temp_tc * vd_wk1) *
                        ((delz * (double)rho_phy[lv]) /
                         ((double)dz8w[lb] * (double)rho_phy[lb]));
                }
            }
        }
        graset[(size_t)b * ncol + col] = gset;
        grasetvel[(size_t)b * ncol + col] = gvel;
    }
}

// ---------------------------------------------------------------------------
// woof-own: aerodynamic resistance and the surface deposition tendency
// ---------------------------------------------------------------------------

// R_a = (ln(z_ref/z0) - psi_h(z_ref/L)) / (kappa u*), held >= 0, real(4).
// psi_h is Businger-Dyer (Paulson 1970): -5 zeta for zeta >= 0, else
// 2 ln((1 + sqrt(1 - 16 zeta)) / 2).  An infinite L is neutral.
extern "C" __global__ void smoke_deposition_resist_a(
    const float *__restrict__ ustar, const float *__restrict__ z0,
    const float *__restrict__ z_ref, const float *__restrict__ obukhov,
    float *__restrict__ resist_a, int n)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    // zeta held at 1, the top of the stable range the Businger-Dyer form
    // was fitted over (very stable cells would otherwise give R_a ~ 1e4).
    const float zeta = fminf(z_ref[c] / obukhov[c], 1.0f);
    float psi;
    if (zeta >= 0.0f) {
        psi = -5.0f * zeta;
    } else {
        psi = 2.0f * gfk_log((1.0f + sqrtf(1.0f - 16.0f * zeta)) * 0.5f);
    }
    const float ra = (gfk_log(z_ref[c] / z0[c]) - psi) / (0.4f * ustar[c]);
    resist_a[c] = ra > 0.0f ? ra : 0.0f;
}

// dq0/dt = -(v_d q0) / dz0 and F = (rho0 v_d) q0 (downward), real(4).
extern "C" __global__ void smoke_deposition_surface_tendency(
    const float *__restrict__ q0, const float *__restrict__ dz0,
    const float *__restrict__ v_d, const float *__restrict__ rho0,
    float *__restrict__ tendency, float *__restrict__ flux, int n)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    tendency[c] = -(v_d[c] * q0[c]) / dz0[c];
    flux[c] = (rho0[c] * v_d[c]) * q0[c];
}
