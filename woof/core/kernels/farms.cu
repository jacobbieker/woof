// WRF-Solar FARMS: the Fast All-sky Radiation Model for Solar applications.
//
// Statement for statement WRF v4.7.1 phys/module_ra_farms.F (commit
// f52c197ed, sha256 1b9e060d...; byte-identical in v4.6.1), the routine
// module_radiation_driver.F calls on EVERY model step under swint_opt = 2
// (:2983-3101, after ENDIF Radiation_step):
//
//   Farms_driver   per column: the night zeros; the dry-air layer mass
//                  rho*dz8w/(1+qv); the radii fallbacks (has_reqc/i/s);
//                  the water vapour, liquid, ice and snow paths and the
//                  path-weighted radii and cloud fraction (Integrate_1var /
//                  Integrate_2var, which stop at kte - 1); the liquid and
//                  ice/snow optical depths (Matrosov et al. 2002, eq. 10
//                  for ice); the Angstrom turbidity beta at 1000 nm
//   FARMS          SUNEARTH, CLEARSKYALL (BIRD at ten zenith angles for the
//                  upward clear-sky transmittance, then at the sun),
//                  WATERMODEL / ICEMODEL for each phase present, the
//                  optical-depth weighted cloud transmittance and
//                  reflectance, the all-sky and clear-sky irradiances, and
//                  the horizontal cloud-fraction blend of GHI and DNI
//
// One thread per column.  Inputs are the driver's own words: the radii in
// metres (WRF's re_cloud/re_ice/re_snow), the 3-D fields [k * kstride +
// column] over the nz mass layers (kts..kte), the surface full-level
// pressure p8w(i, kts, j) per column, and juday = INT(julian).
//
// Numerics.  Every add, subtract, multiply and divide is an explicit IEEE
// single-rounded intrinsic (FADD and friends, glibc_flt32.cuh), so NVRTC
// neither contracts a multiply-add nor rewrites a division as a reciprocal
// multiply; the module also compiles with --fmad=false and no FTZ.  Every
// transcendental is WOOF's own binary32 routine, the words the oracle links
// in place of the C library (tools/farms_wrf471_oracle/libm_ports.cpp):
// gfk_exp / gfk_pow / gfk_log10 (glibc 2.39), glibc_sinf / glibc_cosf /
// glibc_acosf (glibc 2.43) and gfk_sinh (correctly rounded, sinhf_flt32.cuh).
// gfortran -O0 lowers every REAL ** REAL, x**2.0 included, to powf, so those
// are gfk_pow here too.  tests/test_farms_wrf471_parity.py grades every
// output word against the compiled Fortran at 0 ULP.

#define FARMS_PI 3.14159274101257324f        // acos(-1.0), folded: 0x40490fdb
#define FARMS_DEGRAD 0.017453292f            // REAL(.017453293d+00): 0x3c8efa35

__device__ __forceinline__ float farms_max(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ float farms_min(float a, float b) { return a < b ? a : b; }

// BIRD (Bird 1981) at zenith angle z (degrees): Tddclr, Tduclr, Ruuclr.
// ssa and Radius are arguments the Fortran never reads.
__device__ void farms_bird(float p, float albdo, float g, float z, float beta,
                           float alpha, float ozone, float w,
                           float* tddclr, float* tduclr, float* ruuclr)
{
    const float cz = glibc_cosf(FMUL(z, FARMS_DEGRAD));
    const float airmass = FDIV(1.0f, FADD(cz, FMUL(0.15f,
        gfk_pow(FSUB(93.885f, z), -1.25f))));
    const float airmassp = FDIV(FMUL(p, airmass), 1013.0f);
    const float t_rayleigh = gfk_exp(-FMUL(FMUL(0.0903f,
        gfk_pow(airmassp, 0.84f)),
        FSUB(FADD(1.0f, airmassp), gfk_pow(airmassp, 1.01f))));
    const float x0 = FMUL(ozone, airmass);
    const float t_o = FSUB(
        FSUB(1.0f, FMUL(FMUL(0.1611f, x0),
                        gfk_pow(FADD(1.0f, FMUL(139.48f, x0)), -0.3035f))),
        FDIV(FMUL(0.002715f, x0),
             FADD(FADD(1.0f, FMUL(0.044f, x0)),
                  FMUL(0.0003f, gfk_pow(x0, 2.0f)))));
    const float t_gases = gfk_exp(-FMUL(0.0127f, gfk_pow(airmassp, 0.26f)));
    const float xw = FMUL(w, airmass);
    const float t_water = FSUB(1.0f, FDIV(FMUL(2.4959f, xw),
        FADD(gfk_pow(FADD(1.0f, FMUL(79.034f, xw)), 0.6828f),
             FMUL(6.385f, xw))));
    const float tau038 = FMUL(beta, gfk_pow(0.38f, -alpha));
    const float tau050 = FMUL(beta, gfk_pow(0.50f, -alpha));
    const float taua = FADD(FMUL(0.2758f, tau038), FMUL(0.35f, tau050));
    const float t_aerosol = gfk_exp(-FMUL(FMUL(gfk_pow(taua, 0.873f),
        FSUB(FADD(1.0f, taua), gfk_pow(taua, 0.7088f))),
        gfk_pow(airmass, 0.9108f)));
    const float t_aa = FSUB(1.0f, FMUL(FMUL(0.1f,
        FADD(FSUB(1.0f, airmass), gfk_pow(airmass, 1.06f))),
        FSUB(1.0f, t_aerosol)));
    const float t_as = FDIV(t_aerosol, t_aa);
    *ruuclr = FADD(0.0685f, FMUL(FSUB(1.0f, g), FSUB(1.0f, t_as)));
    const float f0 = 1.0f;
    float fddclr = FMUL(f0, cz);
    fddclr = FMUL(fddclr, 0.9662f);
    fddclr = FMUL(fddclr, t_rayleigh);
    fddclr = FMUL(fddclr, t_o);
    fddclr = FMUL(fddclr, t_gases);
    fddclr = FMUL(fddclr, t_water);
    fddclr = FMUL(fddclr, t_aerosol);
    float fdif = FMUL(f0, cz);
    fdif = FMUL(fdif, 0.79f);
    fdif = FMUL(fdif, t_o);
    fdif = FMUL(fdif, t_gases);
    fdif = FMUL(fdif, t_water);
    fdif = FMUL(fdif, t_aa);
    fdif = FMUL(fdif, FADD(FMUL(0.5f, FSUB(1.0f, t_rayleigh)),
                           FMUL(g, FSUB(1.0f, t_as))));
    fdif = FDIV(fdif, FADD(FSUB(1.0f, airmass), gfk_pow(airmass, 1.02f)));
    const float ftotal = FDIV(FADD(fddclr, fdif),
                              FSUB(1.0f, FMUL(albdo, *ruuclr)));
    *tddclr = FDIV(fddclr, FMUL(f0, cz));
    *tduclr = FDIV(FSUB(ftotal, fddclr), FMUL(f0, cz));
}

// WATERMODEL (ice = 0) and ICEMODEL (ice = 1): Tducld, Ruucld.
__device__ void farms_cloud(int ice, float tau, float de, float mu,
                            float* tducld, float* ruucld)
{
    const float solarconst = 1385.72180f;
    float ptau, pdhi, delta, y;
    if (!ice) {
        ptau = FSUB(FMUL(FADD(2.8850f, FMUL(0.002f, FSUB(de, 60.0f))), mu),
                    0.007347f);
        ptau = farms_max(ptau, 0.000001f);
        pdhi = FMUL(FMUL(FADD(1.0f, FMUL(FSUB(de, 60.0f), 0.0002f)), 1087.24f),
                    gfk_pow(mu, 1.1605f));
        delta = FADD(-FMUL(0.644531f, mu), 1.20117f);
        delta = FADD(delta, FDIV(0.129807f, mu));
        delta = FSUB(delta, FDIV(0.00121096f, FMUL(mu, mu)));
        delta = FADD(delta, FDIV(1.52587e-07f, FMUL(FMUL(mu, mu), mu)));
        delta = farms_max(delta, 0.000001f);
        y = FMUL(FMUL(0.012f, FSUB(tau, ptau)), mu);
    } else {
        // De is finite here (a clamped path radius), so exactly one of the
        // Fortran's two IFs assigns Ptau.
        if (de <= 26.0f) {
            ptau = FSUB(FMUL(2.8487f, mu), 0.0029f);
        } else {
            ptau = FSUB(FMUL(FADD(2.8355f, FMUL(FSUB(100.0f, de), 0.006f)), mu),
                        0.00612f);
        }
        ptau = farms_max(ptau, 0.000001f);
        pdhi = FMUL(1047.6367f, gfk_pow(mu, 1.0883f));
        delta = FADD(-FMUL(0.0549531f, mu), 0.617632f);
        delta = FADD(delta, FDIV(0.17876f, mu));
        delta = FSUB(delta, FDIV(0.002174f, FMUL(mu, mu)));
        delta = farms_max(delta, 0.000001f);
        y = FMUL(FMUL(0.01f, FSUB(tau, ptau)), mu);
    }
    const float ltau = gfk_log10(tau);
    const float dl = FSUB(ltau, gfk_log10(ptau));
    const float ppdhi = FMUL(FMUL(FADD(1.0f, gfk_sinh(y)), pdhi),
                             gfk_exp(-FDIV(gfk_pow(dl, 2.0f), delta)));
    *tducld = FDIV(ppdhi, FMUL(solarconst, mu));
    if (tau < 1.0f) {
        *ruucld = FMUL(ice ? 0.094039f : 0.107359f, tau);
    } else {
        const float a = FADD(0.5f, ltau);
        const float b = FADD(0.5f, gfk_log10(tau));
        *ruucld = FSUB(ice ? 1.02f : 1.03f,
                       gfk_exp(-FDIV(FMUL(a, b), ice ? 3.25f : 3.105f)));
    }
}

extern "C" __global__ void farms_columns(
    int ncol, int nz, long long kstride, int has_reqc, int has_reqi,
    int has_reqs, int juday,
    const float* __restrict__ p8w_sfc, const float* __restrict__ albedo,
    const float* __restrict__ coszen, const float* __restrict__ aerssa,
    const float* __restrict__ aerasy, const float* __restrict__ aod550,
    const float* __restrict__ angexp,
    const float* __restrict__ rho, const float* __restrict__ dz8w,
    const float* __restrict__ qv, const float* __restrict__ qc,
    const float* __restrict__ qi, const float* __restrict__ qs,
    const float* __restrict__ re_cloud, const float* __restrict__ re_ice,
    const float* __restrict__ re_snow, const float* __restrict__ cldfra,
    float* __restrict__ swdown2, float* __restrict__ swddir2,
    float* __restrict__ swddni2, float* __restrict__ swddif2,
    float* __restrict__ swdownc2, float* __restrict__ swddnic2)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    const float mu = coszen[i];
    if (mu <= 0.0f) {
        swdown2[i] = 0.0f; swddni2[i] = 0.0f; swddir2[i] = 0.0f;
        swddif2[i] = 0.0f; swdownc2[i] = 0.0f; swddnic2[i] = 0.0f;
        return;
    }
    const float RE_CLOUD_CLIM = 8.E-6f, RE_ICE_CLIM = 24.E-6f,
                RE_SNOW_CLIM = 24.E-6f;
    // ---- Farms_driver: the column integrals over kts .. kte-1 ----------
    float pmw = 0.0f;
    float lwp = 0.0f, iwp = 0.0f, swp = 0.0f;
    float rc_sum = 0.0f, ri_sum = 0.0f, rs_sum = 0.0f;
    float qt_sum = 0.0f, cf_sum = 0.0f;
    for (int k = 0; k < nz - 1; ++k) {
        const long long at = (long long)k * kstride + i;
        const float rhodz = FDIV(FMUL(rho[at], dz8w[at]), FADD(1.0f, qv[at]));
        const float cf = cldfra[at];
        float rc = re_cloud[at], ri = re_ice[at], rs = re_snow[at];
        if (has_reqc == 1) {
            if (cf > 0.0f && rc < 2.5E-6f) rc = RE_CLOUD_CLIM;
        } else {
            rc = RE_CLOUD_CLIM;
        }
        if (has_reqi == 1) {
            if (cf > 0.0f && ri < 5.0E-6f) ri = RE_ICE_CLIM;
        } else {
            ri = RE_ICE_CLIM;
        }
        if (has_reqs == 1) {
            if (cf > 0.0f && rs < 10.0E-6f) rs = RE_SNOW_CLIM;
        } else {
            rs = RE_SNOW_CLIM;
        }
        const float c = qc[at], ice = qi[at], snow = qs[at];
        pmw = FADD(pmw, FMUL(qv[at], rhodz));
        lwp = FADD(lwp, FMUL(c, rhodz));
        rc_sum = FADD(rc_sum, FMUL(FMUL(c, rc), rhodz));
        iwp = FADD(iwp, FMUL(ice, rhodz));
        ri_sum = FADD(ri_sum, FMUL(FMUL(ice, ri), rhodz));
        swp = FADD(swp, FMUL(snow, rhodz));
        rs_sum = FADD(rs_sum, FMUL(FMUL(snow, rs), rhodz));
        const float qt = FADD(FADD(c, ice), snow);
        qt_sum = FADD(qt_sum, FMUL(qt, rhodz));
        cf_sum = FADD(cf_sum, FMUL(FMUL(qt, cf), rhodz));
    }
    const float re_cloud_path = lwp > 0.0f ? FDIV(rc_sum, lwp) : 0.0f;
    const float re_ice_path = iwp > 0.0f ? FDIV(ri_sum, iwp) : 0.0f;
    const float re_snow_path = swp > 0.0f ? FDIV(rs_sum, swp) : 0.0f;
    const float cldfra_h = qt_sum > 0.0f ? FDIV(cf_sum, qt_sum) : 0.0f;
    const float tau_qv = re_cloud_path > 0.0f
        ? FDIV(FDIV(FMUL(1.5f, lwp), re_cloud_path), 1000.0f) : 0.0f;
    const float tau_qi = re_ice_path > 0.0f
        ? FMUL(FMUL(iwp, 1000.0f), FADD(0.02f,
              FDIV(4.2f, FMUL(FMUL(2.0f, re_ice_path), 1.0e+6f))))
        : 0.0f;
    const float tau_qs = re_snow_path > 0.0f
        ? FMUL(FMUL(swp, 1000.0f), FADD(0.02f,
              FDIV(4.2f, FMUL(FMUL(2.0f, re_snow_path), 1.0e+6f))))
        : 0.0f;
    const float alpha = angexp[i];
    const float beta = FMUL(aod550[i], gfk_pow(FDIV(1000.0f, 550.0f), -alpha));
    const float albdo = albedo[i];
    const float g = aerasy[i];
    (void)aerssa;  // FARMS passes SSA to BIRD, which never reads it

    // ---- FARMS --------------------------------------------------------
    const float p = FDIV(p8w_sfc[i], 100.0f);
    const float ozone = 0.265f;
    const float w = FDIV(pmw, 10.0f);
    float de_cloud2 = farms_max(FDIV(FMUL(2.0f, re_cloud_path), 1.0e-6f), 5.0f);
    de_cloud2 = farms_min(de_cloud2, 120.0f);
    float de_ice2 = farms_max(FDIV(FMUL(2.0f, re_ice_path), 1.0e-6f), 5.0f);
    de_ice2 = farms_min(de_ice2, 140.0f);
    float de_snow2 = farms_max(FDIV(FMUL(2.0f, re_snow_path), 1.0e-6f), 5.0f);
    de_snow2 = farms_min(de_snow2, 140.0f);
    const float TAU_MIN = 0.0001f, TAU_MAX = 300.0f;
    float tau_qv2 = farms_min(farms_max(tau_qv, TAU_MIN), TAU_MAX);
    float tau_qi2 = farms_min(farms_max(tau_qi, TAU_MIN), TAU_MAX);
    float tau_qs2 = farms_min(farms_max(tau_qs, TAU_MIN), TAU_MAX);
    float tau_tot = 0.0f;
    if (tau_qv > TAU_MIN) tau_tot = FADD(tau_tot, tau_qv2);
    if (tau_qi > TAU_MIN) tau_tot = FADD(tau_tot, tau_qi2);
    if (tau_qs > TAU_MIN) tau_tot = FADD(tau_tot, tau_qs2);
    tau_tot = farms_min(tau_tot, TAU_MAX);

    const float z = FDIV(FMUL(glibc_acosf(mu), 180.0f), FARMS_PI);

    // SUNEARTH
    const float b = FDIV(FMUL(FMUL(2.0f, FARMS_PI), (float)juday), 365.0f);
    const float two_b = FMUL(2.0f, b);
    float r1 = FADD(1.00011f, FMUL(0.034221f, glibc_cosf(b)));
    r1 = FADD(r1, FMUL(0.001280f, glibc_sinf(b)));
    r1 = FADD(r1, FMUL(0.000719f, glibc_cosf(two_b)));
    r1 = FADD(r1, FMUL(0.000077f, glibc_sinf(two_b)));
    const float radius = gfk_pow(r1, -0.5f);
    const float f0 = FDIV(1361.2f, FMUL(radius, radius));

    // CLEARSKYALL (USE_REST2 = .false.: BIRD)
    float tddclr, tduclr, ruuclr;
    float tuuclr = 0.0f;
    for (int n = 1; n <= 10; ++n) {
        const float mun = FADD(FMUL(FSUB((float)n, 1.0f), 0.1f), 0.1f);
        const float angle = FDIV(FMUL(glibc_acosf(mun), 180.0f), FARMS_PI);
        farms_bird(p, albdo, g, angle, beta, alpha, ozone, w,
                   &tddclr, &tduclr, &ruuclr);
        tuuclr = FADD(tuuclr, FMUL(FMUL(FMUL(2.0f, mun), tddclr), 0.1f));
    }
    farms_bird(p, albdo, g, z, beta, alpha, ozone, w,
               &tddclr, &tduclr, &ruuclr);

    float tducld_water = 0.0f, ruucld_water = 0.0f;
    float tducld_ice = 0.0f, ruucld_ice = 0.0f;
    float tducld_snow = 0.0f, ruucld_snow = 0.0f;
    if (tau_qv > TAU_MIN) {
        farms_cloud(0, tau_tot, de_cloud2, mu, &tducld_water, &ruucld_water);
    } else {
        tau_qv2 = 0.0f;
    }
    if (tau_qi > TAU_MIN) {
        farms_cloud(1, tau_tot, de_ice2, mu, &tducld_ice, &ruucld_ice);
    } else {
        tau_qi2 = 0.0f;
    }
    if (tau_qs > TAU_MIN) {
        farms_cloud(1, tau_tot, de_snow2, mu, &tducld_snow, &ruucld_snow);
    } else {
        tau_qs2 = 0.0f;
    }
    float tducld, ruucld;
    if (tau_tot > 0.0f) {
        tducld = FDIV(FADD(FADD(FMUL(tau_qv2, tducld_water),
                                FMUL(tau_qi2, tducld_ice)),
                           FMUL(tau_qs2, tducld_snow)), tau_tot);
        tducld = farms_max(farms_min(tducld, 1.0f), 0.0f);
        ruucld = FDIV(FADD(FADD(FMUL(tau_qv2, ruucld_water),
                                FMUL(tau_qi2, ruucld_ice)),
                           FMUL(tau_qs2, ruucld_snow)), tau_tot);
        ruucld = farms_max(farms_min(ruucld, 1.0f), 0.0f);
    } else {
        tducld = 0.0f;
        ruucld = 0.0f;
    }
    float tau = FADD(FADD(tau_qv2, tau_qi2), tau_qs2);
    tau = farms_max(farms_min(tau, TAU_MAX), TAU_MIN);
    const float tddcld = gfk_exp(FDIV(-tau, mu));

    const float dni_clear = FMUL(f0, tddclr);
    float dni = FMUL(dni_clear, tddcld);
    const float dir = FMUL(dni, mu);
    const float f1 = FMUL(FMUL(mu, f0),
        FADD(FMUL(tddcld, FADD(tddclr, tduclr)), FMUL(tducld, tuuclr)));
    const float ftotal = FDIV(f1, FSUB(1.0f, FMUL(albdo,
        FADD(ruuclr, FMUL(FMUL(ruucld, tuuclr), tuuclr)))));
    const float ghi_clear = FMUL(FMUL(mu, f0),
        FDIV(FADD(tddclr, tduclr), FSUB(1.0f, FMUL(albdo, ruuclr))));
    const float one_minus = FSUB(1.0f, cldfra_h);
    const float ghi = FADD(FMUL(cldfra_h, ftotal), FMUL(one_minus, ghi_clear));
    dni = FADD(FMUL(cldfra_h, dni), FMUL(one_minus, dni_clear));
    swdown2[i] = ghi;
    swddni2[i] = dni;
    swddif2[i] = FSUB(ghi, dir);
    swddir2[i] = dir;
    swdownc2[i] = ghi_clear;
    swddnic2[i] = dni_clear;
}
