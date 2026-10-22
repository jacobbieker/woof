// Surface fire spread: WRF v4.7.1 fire physics (float32) and the full
// Rothermel (1972) / Albini (1976) model (float64), one thread per cell.
//
// Compiled through NVRTC directly with -fmad=false --ftz=false
// (woof/fire/_device.py), with glibc_flt32.cuh and glibc_flt64.cuh
// prepended, so every transcendental is glibc's own word and no product is
// fused.  Each kernel mirrors, operation for operation, a NumPy reference:
//
//   wrf_fire_params      woof.fire.wrf_sfire._set_fire_params_host
//                        <- phys/module_fr_fire_phys.F:set_fire_params
//   wrf_fire_ros         woof.fire.wrf_sfire._fire_ros_host (float)
//                        <- phys/module_fr_fire_phys.F:fire_ros
//   wrf_fire_ros_total   woof.fire.wrf_sfire.fire_ros_total
//                        <- phys/module_fr_fire_core.F:2049-2051
//   rothermel_fire_ros   the same transcription in double, for the full
//                        model's coefficients
//   rothermel_head_fire  woof.fire.rothermel._fuel_bed_host + _head_fire_host
//   rothermel_coefficients  woof.fire.rothermel._fuel_bed_host and the NumPy
//                        branch of rothermel_fire_params
//
// WRF is real(4): float arithmetic is spelled with the _rn intrinsics so no
// NVVM rewrite (reciprocal division, contraction) can reach it, and a
// Fortran ``x**2`` is ``x*x``, ``x**3`` is ``(x*x)*x`` as gfortran expands
// a constant integer power.  The templated fire_ros spells its literals in
// double: R(196.850) is the same float as Fortran's real(4) 196.850 for
// every literal there (tests/test_fire_ros_wrf471_parity.py checks each).
// Fortran MAX(a, b) / MIN(a, b) are gfortran's
// ``b > a ? b : a`` / ``b < a ? b : a`` (fr_max / fr_min): the first
// argument wins a tie, which decides the sign of a zero.

#define F_ADD(a, b) __fadd_rn((a), (b))
#define F_SUB(a, b) __fsub_rn((a), (b))
#define F_MUL(a, b) __fmul_rn((a), (b))
#define F_DIV(a, b) __fdiv_rn((a), (b))

__device__ __forceinline__ float fr_max(float a, float b) { return b > a ? b : a; }
__device__ __forceinline__ float fr_min(float a, float b) { return b < a ? b : a; }

// --------------------------------------------------------------------------
// WRF v4.7.1 set_fire_params, per cell.  tab is field-major over the table
// rows: tab[f * nrow + row] with f = 0 fgi, 1 fgi_lh, 2 fueldepthm, 3 savr,
// 4 fuelmce, 5 fueldens, 6 st, 7 se, 8 weight, 9 ichap.  row < 0 is the
// no-fuel category.
// --------------------------------------------------------------------------
extern "C" __global__ void wrf_fire_params(
    const int* __restrict__ row_of, const float* __restrict__ fmc_g, long n,
    const float* __restrict__ tab, int nrow, float fuelheat, float fuelmc_g_lh,
    float* __restrict__ fuel_time, float* __restrict__ fgip,
    float* __restrict__ ischap, float* __restrict__ betafl,
    float* __restrict__ bbb, float* __restrict__ phiwc,
    float* __restrict__ r_0, float* __restrict__ iboros)
{
    const long c = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const int k = row_of[c];
    if (k < 0) {
        fgip[c] = 0.0f;
        ischap[c] = 0.0f;
        betafl[c] = 1.0f;
        bbb[c] = 1.0f;
        fuel_time[c] = F_DIV(7.0f, 0.85f);
        phiwc[c] = 0.0f;
        r_0[c] = 0.0f;
        iboros[c] = 0.0f;
        return;
    }
#define T(f) tab[(f) * nrow + k]
    const float fgi = T(0), fgi_lh = T(1), fueldepthm = T(2), savr = T(3);
    const float fuelmce = T(4), fueldens = T(5), st = T(6), se = T(7);
    const float weight = T(8), ichap = T(9);
#undef T
    const float fmc = fmc_g[c];
    fuel_time[c] = F_DIV(weight, 0.85f);
    ischap[c] = ichap;
    float fg;
    if (fuelmc_g_lh > 0.3f && fuelmc_g_lh < 1.2f) {
        fg = F_ADD(fgi, F_MUL(F_SUB(1.0f, F_DIV(F_SUB(fuelmc_g_lh, 0.3f), 0.9f)),
                              fgi_lh));
    } else if (fuelmc_g_lh <= 0.3f) {
        fg = F_ADD(fgi, fgi_lh);
    } else {
        fg = fgi;
    }
    fgip[c] = fg;

    const float bmst = F_DIV(fmc, F_ADD(1.0f, fmc));
    const float fuelloadm = F_MUL(F_SUB(1.0f, bmst), fg);
    // (.3048)**2 folds to the float square of 0.3048f at compile time.
    const float ft2 = F_MUL(0.3048f, 0.3048f);
    const float fuelload = F_MUL(F_MUL(fuelloadm, ft2), 2.205f);
    const float fueldepth = F_DIV(fueldepthm, 0.3048f);
    const float bf = F_DIV(fuelload, F_MUL(fueldepth, fueldens));
    const float betaop = F_MUL(3.348f, gfk_pow(savr, -0.8189f));
    const float qig = F_ADD(250.0f, F_MUL(1116.0f, fmc));
    const float epsilon = gfk_exp(F_DIV(-138.0f, savr));
    const float rhob = F_DIV(fuelload, fueldepth);
    const float cc = F_MUL(7.47f, gfk_exp(F_MUL(-0.133f, gfk_pow(savr, 0.55f))));
    const float b = F_MUL(0.02526f, gfk_pow(savr, 0.54f));
    const float e = F_MUL(0.715f, gfk_exp(F_MUL(-3.59e-4f, savr)));
    const float pw = F_MUL(cc, gfk_pow(F_DIV(bf, betaop), -e));
    const float rtemp2 = gfk_pow(savr, 1.5f);
    const float gammax = F_DIV(rtemp2, F_ADD(495.0f, F_MUL(0.0594f, rtemp2)));
    const float a = F_DIV(1.0f, F_SUB(F_MUL(4.774f, gfk_pow(savr, 0.1f)), 7.27f));
    const float ratio = F_DIV(bf, betaop);
    const float gamma = F_MUL(F_MUL(gammax, gfk_pow(ratio, a)),
                              gfk_exp(F_MUL(a, F_SUB(1.0f, ratio))));
    const float wn = F_DIV(fuelload, F_ADD(1.0f, st));
    const float r1 = F_DIV(fmc, fuelmce);
    const float r1sq = F_MUL(r1, r1);
    const float etam = F_SUB(F_ADD(F_SUB(1.0f, F_MUL(2.59f, r1)), F_MUL(5.11f, r1sq)),
                             F_MUL(3.52f, F_MUL(r1sq, r1)));
    const float etas = F_MUL(0.174f, gfk_pow(se, -0.19f));
    const float ir = F_MUL(F_MUL(F_MUL(F_MUL(gamma, wn), fuelheat), etam), etas);
    // iboros = ir * 1055./(0.3048**2 * 60.) * 1.e-3 * (60.*12.6/savr)
    const float k1 = F_MUL(ft2, 60.0f);
    const float k2 = F_MUL(60.0f, 12.6f);
    iboros[c] = F_MUL(F_MUL(F_DIV(F_MUL(ir, 1055.0f), k1), 1.e-3f), F_DIV(k2, savr));
    const float xifr = F_DIV(
        gfk_exp(F_MUL(F_ADD(0.792f, F_MUL(0.681f, gfk_pow(savr, 0.5f))), F_ADD(bf, 0.1f))),
        F_ADD(192.0f, F_MUL(0.2595f, savr)));
    r_0[c] = F_DIV(F_MUL(ir, xifr), F_MUL(F_MUL(rhob, epsilon), qig));
    betafl[c] = bf;
    bbb[c] = b;
    phiwc[c] = pw;
}

// --------------------------------------------------------------------------
// WRF v4.7.1 fire_ros (ibeh = 1), per node, float and double.
// --------------------------------------------------------------------------
template <typename R> struct fr_ops;
template <> struct fr_ops<float> {
    __device__ static float add(float a, float b) { return __fadd_rn(a, b); }
    __device__ static float sub(float a, float b) { return __fsub_rn(a, b); }
    __device__ static float mul(float a, float b) { return __fmul_rn(a, b); }
    __device__ static float div(float a, float b) { return __fdiv_rn(a, b); }
    __device__ static float sqrt_(float a) { return __fsqrt_rn(a); }
    __device__ static float pow_(float a, float b) { return gfk_pow(a, b); }
    __device__ static float tiny() { return 1.17549435e-38f; }
};
template <> struct fr_ops<double> {
    __device__ static double add(double a, double b) { return __dadd_rn(a, b); }
    __device__ static double sub(double a, double b) { return __dsub_rn(a, b); }
    __device__ static double mul(double a, double b) { return __dmul_rn(a, b); }
    __device__ static double div(double a, double b) { return __ddiv_rn(a, b); }
    __device__ static double sqrt_(double a) { return __dsqrt_rn(a); }
    __device__ static double pow_(double a, double b) { return glibc_pow(a, b); }
    __device__ static double tiny() { return 2.2250738585072014e-308; }
};

template <typename R>
__device__ __forceinline__ R fr_maxt(R a, R b) { return b > a ? b : a; }
template <typename R>
__device__ __forceinline__ R fr_mint(R a, R b) { return b < a ? b : a; }

template <typename R>
__device__ void fire_ros_t(R r_0, R phiwc, R bbb, R betafl, R ischap,
                           R vx, R vy, R dzdx, R dzdy, R nvx, R nvy,
                           int advection, R ros_max,
                           R& ros_base, R& ros_wind, R& ros_slope)
{
    typedef fr_ops<R> O;
    const R zero = R(0);
    R speed, tanphi, cor_wind, cor_slope;
    if (advection != 0) {
        speed = O::add(O::sqrt_(O::add(O::mul(vx, vx), O::mul(vy, vy))), O::tiny());
        tanphi = O::add(O::sqrt_(O::add(O::mul(dzdx, dzdx), O::mul(dzdy, dzdy))), O::tiny());
        cor_wind = fr_maxt<R>(zero, O::div(O::add(O::mul(vx, nvx), O::mul(vy, nvy)), speed));
        cor_slope = fr_maxt<R>(zero, O::div(O::add(O::mul(dzdx, nvx), O::mul(dzdy, nvy)), tanphi));
    } else {
        speed = O::add(O::mul(vx, nvx), O::mul(vy, nvy));
        tanphi = O::add(O::mul(dzdx, nvx), O::mul(dzdy, nvy));
        cor_wind = R(1);
        cor_slope = R(1);
    }
    R rb = ros_base, rw, rs;
    if (!(ischap > zero)) {
        const R spdms = fr_maxt<R>(speed, zero);
        const R umidm = fr_mint<R>(spdms, R(30));
        const R umid = O::mul(umidm, R(196.850));
        const R phiw = O::mul(O::pow_(umid, bbb), phiwc);
        R phis = zero;
        if (tanphi > zero) {
            phis = O::mul(O::mul(R(5.275), O::pow_(betafl, R(-0.3))),
                          O::mul(tanphi, tanphi));
        }
        rb = O::mul(r_0, R(.00508));
        rw = O::mul(rb, phiw);
        rs = O::mul(rb, phis);
    } else {
        // ros_base is not assigned on this branch in WRF v4.7.1; the
        // caller's value stands (the port passes 0).
        const R spdms = fr_maxt<R>(speed, zero);
        const R ros_back = R(.03333);
        rw = O::mul(R(1.2974), O::pow_(spdms, R(1.41)));
        rw = fr_maxt<R>(rw, ros_back);
        rs = zero;
    }
    rw = O::mul(rw, cor_wind);
    rs = O::mul(rs, cor_slope);
    const R excess = O::sub(O::add(O::add(rb, rw), rs), ros_max);
    if (excess > zero) {
        rw = O::sub(rw, O::div(O::mul(excess, rw), O::add(rw, rs)));
        rs = O::sub(rs, O::div(O::mul(excess, rs), O::add(rw, rs)));
    }
    ros_base = rb;
    ros_wind = rw;
    ros_slope = rs;
}

#define FIRE_ROS_KERNEL(NAME, R)                                              \
extern "C" __global__ void NAME(                                              \
    const R* __restrict__ r_0, const R* __restrict__ phiwc,                   \
    const R* __restrict__ bbb, const R* __restrict__ betafl,                  \
    const R* __restrict__ ischap, const R* __restrict__ vx,                   \
    const R* __restrict__ vy, const R* __restrict__ dzdx,                     \
    const R* __restrict__ dzdy, const R* __restrict__ nx,                     \
    const R* __restrict__ ny, long n, int advection, R ros_max,               \
    R* __restrict__ ros_base, R* __restrict__ ros_wind,                       \
    R* __restrict__ ros_slope)                                                \
{                                                                             \
    const long c = (long)blockIdx.x * blockDim.x + threadIdx.x;               \
    if (c >= n) return;                                                       \
    R rb = R(0), rw, rs;                                                      \
    fire_ros_t<R>(r_0[c], phiwc[c], bbb[c], betafl[c], ischap[c], vx[c],      \
                  vy[c], dzdx[c], dzdy[c], nx[c], ny[c], advection, ros_max,  \
                  rb, rw, rs);                                                \
    ros_base[c] = rb;                                                         \
    ros_wind[c] = rw;                                                         \
    ros_slope[c] = rs;                                                        \
}

FIRE_ROS_KERNEL(wrf_fire_ros, float)
FIRE_ROS_KERNEL(rothermel_fire_ros, double)

// module_fr_fire_core.F:2049-2051: rr = ros_base + ros_wind +
// fire_slope_factor*ros_slope, then max(rr, 0.) under fire_grows_only.
extern "C" __global__ void wrf_fire_ros_total(
    const float* __restrict__ rb, const float* __restrict__ rw,
    const float* __restrict__ rs, long n, float slope_factor, int grows_only,
    float* __restrict__ rr)
{
    const long c = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float r = F_ADD(F_ADD(rb[c], rw[c]), F_MUL(slope_factor, rs[c]));
    if (grows_only) r = fr_max(r, 0.0f);
    rr[c] = r;
}

// --------------------------------------------------------------------------
// Full Rothermel-Albini model, float64.  tab is field-major over the
// standard models: tab[f * nmodel + model], f in woof.fire.rothermel
// _FIELDS order.  Sums run in class order 0..5, each from 0.0, exactly as
// the NumPy reference adds them.
// --------------------------------------------------------------------------
struct fr_bed {
    double r0, ir, sigma, bulk_density, beta, rpr, mxl, c, b, e;
    double beta_safe, rpr_safe;
    bool burnable;
};

__device__ __forceinline__ double fr_div(double a, double b) { return __ddiv_rn(a, b); }
__device__ __forceinline__ double fr_cubic_damping(double r)
{
    if (r >= 1.0 - 1e-12) return 0.0;
    return 1.0 - 2.59 * r + 5.11 * (r * r) - 3.52 * ((r * r) * r);
}

__device__ fr_bed fr_fuel_bed(const double* __restrict__ tab, int nm, int k,
                              double m1, double m10, double m100, double mh,
                              double mw, double curing)
{
#define TB(f) tab[(f) * nm + k]
    double cured;
    if (curing < 0.0) {
        cured = mh < 0.30 ? 1.0 : (mh <= 1.20 ? 1.333 - 1.11 * mh : 0.0);
    } else {
        cured = curing;
    }
    if (!(TB(17) > 0.5)) cured = 0.0;
    const double wherb = TB(3);
    const double wdh = wherb * cured;
    const double w[6] = {TB(0), TB(1), TB(2), wdh, wherb - wdh, TB(4)};
    const double s[6] = {TB(5), TB(6), TB(7), TB(8), TB(8), TB(9)};
    const double m[6] = {m1, m10, m100, m1, mh, mw};
    const double hd = TB(12), hl = TB(13);
    const double depth = TB(10), mxd = TB(11), rhop = TB(14);
    const double st = TB(15), se = TB(16);
#undef TB
    double area[6], f[6], ex138[6];
    for (int j = 0; j < 6; ++j) {
        area[j] = fr_div(w[j] * s[j], rhop);
        ex138[j] = glibc_exp(fr_div(-138.0, s[j]));
    }
    const double a_dead = ((area[0] + area[1]) + area[2]) + area[3];
    const double a_live = area[4] + area[5];
    for (int j = 0; j < 6; ++j) {
        const double acat = j < 4 ? a_dead : a_live;
        f[j] = acat > 1.0e-7 ? fr_div(area[j], acat) : 0.0;
    }
    const double a_total = a_dead + a_live;
    const double f_dead = a_total > 0.0 ? fr_div(a_dead, a_total) : 0.0;
    const double f_live = a_total > 0.0 ? 1.0 - f_dead : 0.0;

    int sub[6];
    for (int j = 0; j < 6; ++j) {
        const double x = s[j];
        sub[j] = x >= 1200.0 ? 0 : x >= 192.0 ? 1 : x >= 96.0 ? 2
               : x >= 48.0 ? 3 : x >= 16.0 ? 4 : 5;
    }
    double wn_d = 0.0, wn_l = 0.0, h_d = 0.0, h_l = 0.0, se_d = 0.0, se_l = 0.0;
    double m_d = 0.0, m_l = 0.0, s_d = 0.0, s_l = 0.0, load = 0.0, bsum = 0.0;
    for (int j = 0; j < 6; ++j) {
        double g = 0.0;
        if (sub[j] < 5) {
            for (int q = 0; q < 6; ++q)
                if (sub[q] == sub[j] && ((q < 4) == (j < 4))) g = g + f[q];
        }
        const double wn = (g * w[j]) * (1.0 - st);
        if (j < 4) {
            wn_d = wn_d + wn; h_d = h_d + f[j] * hd; se_d = se_d + f[j] * se;
            m_d = m_d + f[j] * m[j]; s_d = s_d + f[j] * s[j];
        } else {
            wn_l = wn_l + wn; h_l = h_l + f[j] * hl; se_l = se_l + f[j] * se;
            m_l = m_l + f[j] * m[j]; s_l = s_l + f[j] * s[j];
        }
        load = load + w[j];
        bsum = bsum + fr_div(w[j], rhop);
    }
    const double sig_raw = f_dead * s_d + f_live * s_l;
    fr_bed bed;
    bed.bulk_density = fr_div(load, depth);
    bed.beta = fr_div(bsum, depth);
    bed.burnable = (sig_raw > 1.0e-7) && (load > 0.0);
    const double sig = bed.burnable ? sig_raw : 1.0;
    const double beta_op = 3.348 * glibc_pow(sig, -0.8189);
    const double rpr = fr_div(bed.beta, beta_op);

    double fine_d = 0.0, fine_dm = 0.0, fine_l = 0.0;
    for (int j = 0; j < 4; ++j) {
        const double fw = w[j] * ex138[j];
        fine_d = fine_d + fw;
        fine_dm = fine_dm + fw * m[j];
    }
    for (int j = 4; j < 6; ++j)
        fine_l = fine_l + w[j] * glibc_exp(fr_div(-500.0, s[j]));
    const double fdm = fine_d > 1.0e-7 ? fr_div(fine_dm, fine_d) : 0.0;
    const double ratio_w = fine_l > 1.0e-7 ? fr_div(fine_d, fine_l) : 0.0;
    double mxl = 2.9 * ratio_w * (1.0 - fr_div(fdm, mxd)) - 0.226;
    mxl = mxl > mxd ? mxl : mxd;

    const double etam_d = fr_cubic_damping(fr_div(m_d, mxd));
    const double etam_l = fr_cubic_damping(fr_div(m_l, mxl));
    double etas_d = se_d > 0.0 ? 0.174 * glibc_pow(se_d, -0.19) : 0.0;
    double etas_l = se_l > 0.0 ? 0.174 * glibc_pow(se_l, -0.19) : 0.0;
    etas_d = etas_d < 1.0 ? etas_d : 1.0;
    etas_l = etas_l < 1.0 ? etas_l : 1.0;

    const double sig15 = glibc_pow(sig, 1.5);
    const double gamma_max = fr_div(sig15, 495.0 + 0.0594 * sig15);
    const double a_coef = 133.0 * glibc_pow(sig, -0.7913);
    const double gamma = (gamma_max * glibc_pow(rpr, a_coef))
                       * glibc_exp(a_coef * (1.0 - rpr));
    double ir = gamma * (((wn_d * h_d) * etam_d) * etas_d
                         + ((wn_l * h_l) * etam_l) * etas_l);
    if (!bed.burnable) ir = 0.0;

    const double xi = fr_div(glibc_exp((0.792 + 0.681 * __dsqrt_rn(sig)) * (bed.beta + 0.1)),
                             192.0 + 0.2595 * sig);
    double hs = 0.0;
    for (int j = 0; j < 6; ++j) {
        const double fc = j < 4 ? f_dead : f_live;
        hs = hs + ((fc * f[j]) * ex138[j]) * (250.0 + 1116.0 * m[j]);
    }
    hs = bed.bulk_density * hs;
    double r0 = hs > 1.0e-7 ? fr_div(ir * xi, hs) : 0.0;
    if (!bed.burnable) r0 = 0.0;

    bed.r0 = r0;
    bed.ir = ir;
    bed.sigma = bed.burnable ? sig_raw : 0.0;
    bed.rpr = bed.burnable ? rpr : 0.0;
    bed.rpr_safe = bed.burnable ? rpr : 1.0;
    bed.beta_safe = bed.beta > 0.0 ? bed.beta : 1.0;
    bed.mxl = mxl;
    bed.c = 7.47 * glibc_exp(-0.133 * glibc_pow(sig, 0.55));
    bed.b = 0.02526 * glibc_pow(sig, 0.54);
    bed.e = 0.715 * glibc_exp(-3.59e-4 * sig);
    return bed;
}

// wind_limit: 0 none, 1 Rothermel (1972) U <= 0.9 I_R, 2 Andrews et al.
// (2013) U <= 96.8 I_R^(1/3).  US units in, US units out; the host converts.
extern "C" __global__ void rothermel_head_fire(
    const int* __restrict__ model, const double* __restrict__ m1,
    const double* __restrict__ m10, const double* __restrict__ m100,
    const double* __restrict__ mh, const double* __restrict__ mw,
    const double* __restrict__ curing, const double* __restrict__ wind,
    const double* __restrict__ slope, long n, const double* __restrict__ tab,
    int nmodel, int wind_limit, double* __restrict__ out, long stride,
    unsigned char* __restrict__ limited_out)
{
    const long c = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const fr_bed bed = fr_fuel_bed(tab, nmodel, model[c], m1[c], m10[c],
                                   m100[c], mh[c], mw[c], curing[c]);
    const double wv = wind[c];
    const double u = (wv > 0.0 ? wv : 0.0) * fr_div(60.0, 0.3048);
    const double rp = bed.rpr_safe;
    double phi_w = u >= 1.0e-7
        ? (bed.c * glibc_pow(u, bed.b)) * glibc_pow(rp, -bed.e) : 0.0;
    const double t = fabs(slope[c]);
    double phi_s = (5.275 * glibc_pow(bed.beta_safe, -0.3)) * (t * t);
    if (!bed.burnable) { phi_w = 0.0; phi_s = 0.0; }
    double phi_e = phi_w + phi_s;
    double u_eff = glibc_pow(fr_div(phi_e * glibc_pow(rp, bed.e), bed.c),
                             fr_div(1.0, bed.b));
    bool limited = false;
    if (wind_limit != 0) {
        const double u_lim = wind_limit == 1
            ? 0.9 * bed.ir : 96.8 * glibc_pow(bed.ir, fr_div(1.0, 3.0));
        limited = (u_eff > u_lim) && bed.burnable && (bed.ir > 0.0);
        if (limited) {
            phi_e = (bed.c * glibc_pow(u_lim, bed.b)) * glibc_pow(rp, -bed.e);
            u_eff = u_lim;
        }
    }
    const double ros = bed.r0 * (1.0 + phi_e);
    const double residence = bed.sigma > 1.0e-7 ? fr_div(384.0, bed.sigma) : 0.0;
    const double hpa = bed.ir * residence;
    const double fli = fr_div(hpa * ros, 60.0);
    const double flame = fli < 1.0e-7 ? 0.0 : 0.45 * glibc_pow(fli, 0.46);
    const double values[19] = {
        ros, bed.r0, phi_w, phi_s, u_eff, bed.ir, hpa, fli, flame, residence,
        bed.sigma, bed.bulk_density, bed.beta, bed.rpr, bed.mxl, bed.c, bed.b,
        bed.e, 0.0};
    for (int q = 0; q < 18; ++q) out[q * stride + c] = values[q];
    limited_out[c] = limited ? 1 : 0;
}

// The WRF fire_ros coefficients of the full model: r_0 (ft/min),
// phiwc = C (beta/beta_op)^-E, bbb = B, betafl = beta.
extern "C" __global__ void rothermel_coefficients(
    const int* __restrict__ model, const double* __restrict__ m1,
    const double* __restrict__ m10, const double* __restrict__ m100,
    const double* __restrict__ mh, const double* __restrict__ mw,
    const double* __restrict__ curing, long n, const double* __restrict__ tab,
    int nmodel, double* __restrict__ r_0, double* __restrict__ phiwc,
    double* __restrict__ bbb, double* __restrict__ betafl)
{
    const long c = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const fr_bed bed = fr_fuel_bed(tab, nmodel, model[c], m1[c], m10[c],
                                   m100[c], mh[c], mw[c], curing[c]);
    r_0[c] = bed.r0;
    phiwc[c] = bed.c * glibc_pow(bed.rpr_safe, -bed.e);
    bbb[c] = bed.b;
    betafl[c] = bed.beta_safe;
}
