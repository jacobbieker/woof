// WRF-Solar EPS (multi_perturb = 1): the per-scheme perturbation hooks and
// the perturbation-pattern generator.
//
// ---------------------------------------------------------------------------
// 1. The hooks.  AUTHORITY: WRF v4.7.1 (f52c197ed), the drivers' own
// subroutines
//   FARMS     phys/module_radiation_driver.F    Add_/Remove_multi_perturb_swrad_perturbations
//                                               (:5623-5708), called :3052-3082
//   Deng      phys/module_shallowcu_driver.F    Add_/Remove_multi_perturb_shcu_perturbations
//                                               (:695-765), called :632-672
//   Thompson  phys/module_microphysics_driver.F Add_/Remove_multi_perturb_mp_perturbations
//                                               (:2937-3008), called :1120-1213, :1314-1381
//   icloud=3  phys/module_radiation_driver.F    the t_1d/qv_1d statements (:1472-1478)
// graded word for word by tools/wrfsolar_eps_wrf471_oracle (gfortran -O0
// -ffp-contract=off, x86-64).
//
// Every statement of those routines is one of three forms on one element,
// optionally clamped:
//   op 0  (1.0 + pert*fac) * x      the Add_ statements and the cld3 inputs
//   op 1  x / (1.0 + pert*fac)      Remove_ for FARMS' 2-D fields and Deng
//   op 2  x - pert*fac*tmp          Remove_ for FARMS' and Thompson's 3-D fields
// with ``tmp`` the value Add_ saved (qv_tmp, ...).  Fortran's left-to-right
// order: pert*fac first, then 1.0 + (it), then the product, quotient or
// difference; each rounds once (FADD/FMUL/FDIV/FSUB, --fmad=false).
// Clamps: a lower bound, or a lower bound then ``min(., hi)``:
//   1  max, a tie keeps y       2  max, a tie keeps lo
//   3  min(max(.), hi), the max's tie keeps y     4  ... keeps lo
// The only tie that shows is a signed zero against a zero bound (y = -0.0,
// lo = 0.0; the upper bounds are 1.0).  gfortran -O0 on x86-64 lowers
// MAX/MIN to maxss/minss, which return their source operand on a tie, and
// the operand order is GCC's choice per statement, not the source's
// argument order: Add_'s albedo statement keeps the bound, Remove_'s keeps
// y.  woof/core/wrfsolar_eps.py names each statement's form, as the x86-64
// record shows it (tools/wrfsolar_eps_wrf471_oracle).  eps_max(a, b) is
// ``a > b ? a : b`` and eps_min(a, b) ``a < b ? a : b``, selected as bit
// patterns so that no compiler folds them into an IEEE max/min (Metal's
// does, and fmax(0.0, -0.0) is +0.0).
//
// x[x_off + i] is updated for i < n.  ``pert`` is read at pert_off + i, or
// at pert_off + i % plane when ``pert_planar`` = 1: a pattern that is
// constant in height (STOCHPERT.TBL's 2-D rows) is stored as one level and
// read at every level, the very words WRF's pert3d holds there.  The 2-D
// FARMS fields read the lowest pattern level (Add_'s ``k = 1``).  ``save``
// (has_save = 1) receives x before the update, Add_'s qv_tmp = qv; ``tmp``
// (op 2) is read at i.  Both are caller buffers of n floats.

__device__ __forceinline__ float eps_max(float a, float b)
{
    const unsigned int ua = __float_as_uint(a), ub = __float_as_uint(b);
    return __uint_as_float(a > b ? ua : ub);
}
__device__ __forceinline__ float eps_min(float a, float b)
{
    const unsigned int ua = __float_as_uint(a), ub = __float_as_uint(b);
    return __uint_as_float(a < b ? ua : ub);
}

extern "C" __global__ void eps_apply(
    int n, float* __restrict__ x, int x_off, const float* __restrict__ pert,
    int pert_off, int pert_planar, int plane, float fac, int op, int clamp,
    float lo, float hi, const float* __restrict__ tmp,
    float* __restrict__ save, int has_save)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xi = x[x_off + i];
    if (has_save) save[i] = xi;
    const float p = pert[pert_off + (pert_planar ? (i % plane) : i)];
    const float pf = FMUL(p, fac);
    float y;
    if (op == 0) y = FMUL(FADD(1.0f, pf), xi);
    else if (op == 1) y = FDIV(xi, FADD(1.0f, pf));
    else y = FSUB(xi, FMUL(pf, tmp[i]));
    if (clamp == 1) y = eps_max(lo, y);
    else if (clamp == 2) y = eps_max(y, lo);
    else if (clamp == 3) y = eps_min(eps_max(lo, y), hi);
    else if (clamp == 4) y = eps_min(eps_max(y, lo), hi);
    x[x_off + i] = y;
}

// ---------------------------------------------------------------------------
// 2. The pattern generator: WRF v4.7.1 dyn_em/module_stoch.F (e28c912f...)
// for variable 'R' / 'T' -- SETUP_RAND_PERTURB (:527-842), UPDATE_STOCH
// (:848-953), RAND_PERT_UPDATE (:1098-1349) -- as INITIALIZE_STOCH
// (:273-446) and first_rk_step_part2 (:262-317) call them for each
// STOCHPERT.TBL field, on one tile spanning the domain.
//
// Arrays are WRF's (IL, IK) = (x, y) over the staggered domain, LMAX =
// ide-ids+1 and KMAX = jde-jds+1, stored row-major [IK-1][IL-1] -- the
// Fortran (ids:ide, jds:jde) memory order.  Every float32 operation keeps
// Fortran's order and rounds once; EXP and ALOG are glibc's expf/logf
// (gfk_exp/gfk_log), COS/SIN glibc 2.43's cosf/sinf (glibc_cosf/glibc_sinf),
// SQRT IEEE.
//
// Not WRF's: the Gaussian draws.  WRF's UPDATE_STOCH draws them from the
// compiler's RANDOM_NUMBER stream seeded by date and nens (rand_seed);
// these are counter-based Philox4x32-10 draws through the same polar
// transform and |z| < 3 rejection (gauss_noise :1488-1514), the policy of
// woof/ensemble/stochastic.py.  Everything downstream of the draws is WRF's
// arithmetic (tools/wrfsolar_eps_wrf471_oracle grades it with injected
// draws) except the backward FFT, which is cuFFT or pocketfft rather than
// FFTPACK5 (graded to a tolerance there).

// SETUP_RAND_PERTURB('R'): ALPH and SP_AMP.  One thread: ZGAMMAN is a
// sequential float32 sum in WRF's loop order.  ``zchi`` is caller scratch
// of LMAX*KMAX floats; out[0] = ALPH, out[1] = ZCONSTF0, out[2] = ZGAMMAN.
// The forcing band's caps arrive as INITIALIZE_STOCH passes them: its
// actual KMAXFORCT/LMAXFORCT (the namelist's kmaxforct/lmaxforct) land in
// SETUP's dummies KMAXFORCTH/LMAXFORCTH (the argument lists are permuted),
// and SETUP's own KMAXFORCT starts from the domain size.
extern "C" __global__ void mp_setup_amplitude(
    int lmax, int kmax, float dx, float dy, int itime_step,
    float gridpt_stddev, float l_rand_perturb, float tau_rand_perturb,
    int kminforct, int lminforct, int kmaxforcth, int lmaxforcth,
    float* __restrict__ zchi, float* __restrict__ sp_amp,
    float* __restrict__ out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float RPI = 3.141592653589793f;
    const float RY = FMUL((float)kmax, dy);
    const float RX = FMUL((float)lmax, dx);
    int kmaxforct = (lmax / 2 < kmax / 2 ? lmax / 2 : kmax / 2) - 5;
    int lmaxforct = kmaxforct;
    if (kmaxforct > kmaxforcth) kmaxforct = kmaxforcth;
    if (lmaxforct > lmaxforcth) lmaxforct = lmaxforcth;
    const float kappat = FMUL(l_rand_perturb, l_rand_perturb);
    const float phi = gfk_exp(-FDIV((float)itime_step, tau_rand_perturb));
    const float alph = FSUB(1.f, phi);
    // -2*RPI**2*kappat*(...): Fortran's unary minus negates the whole
    // product, whose factors round left to right.
    const float two_pi2 = FMUL(2.f, FMUL(RPI, RPI));
    const float four_pi2 = FMUL(4.f, FMUL(RPI, RPI));
    const float kmax_hi = FDIV(FADD((float)kmaxforct, 0.5f), RX);
    const float kmin_lo = FDIV(FSUB((float)kminforct, 0.5f), RX);
    const float lmax_hi = FDIV(FADD((float)lmaxforct, 0.5f), RY);
    const float lmin_lo = FDIV(FSUB((float)lminforct, 0.5f), RY);
    for (int i = 0; i < lmax * kmax; ++i) zchi[i] = 0.f;
    float zgamman = 0.f;
    for (int ik = 0; ik <= kmax; ++ik) {          // IK = jds-1, jde
        for (int il = 0; il <= lmax; ++il) {      // IL = ids-1, ide
            const float fk = (float)ik, fl = (float)il;
            const float s = FADD(FDIV(FMUL(FDIV(fk, RY), fk), RY),
                                 FDIV(FMUL(FDIV(fl, RX), fl), RX));
            const float r = FSQRT(s);
            if (((r < kmax_hi) && (r >= kmin_lo)) || ((r < lmax_hi) && (r >= lmin_lo))) {
                if (ik > 0 || il > 0) {
                    // ZCHI(IL+1, IK+1): the band never reaches IL = ide or
                    // IK = jde on a domain WRF accepts (KMAXFORCT <= LMAX/2-5)
                    if (il < lmax && ik < kmax)
                        zchi[ik * lmax + il] = gfk_exp(-FMUL(FMUL(two_pi2, kappat), s));
                    zgamman = FADD(zgamman, gfk_exp(-FMUL(FMUL(four_pi2, kappat), s)));
                }
            }
        }
    }
    zgamman = FMUL(4.0f, zgamman);
    const float zconstf0 = FMUL(gridpt_stddev,
        FSQRT(FDIV(FSUB(1.f, FMUL(phi, phi)), FMUL(2.f, zgamman))));
    // SP_AMP = 0, then the four quadrants (:777-812), 1-based IL, IK.
    for (int IK = 1; IK <= kmax; ++IK) {
        for (int IL = 1; IL <= lmax; ++IL) {
            float a = 0.f;
            if (IL <= lmax / 2 + 1 && IK <= kmax / 2 + 1)
                a = FMUL(zconstf0, zchi[(IK - 1) * lmax + (IL - 1)]);
            if (IL > lmax / 2 + 1 && IK <= kmax / 2 + 1)
                a = FMUL(zconstf0, zchi[(IK - 1) * lmax + (lmax - IL + 1)]);
            if (IK > kmax / 2 + 1 && IL <= lmax / 2)
                a = FMUL(zconstf0, zchi[(kmax - IK + 1) * lmax + (IL - 1)]);
            if (IK > kmax / 2 + 1 && IL > lmax / 2)
                a = FMUL(zconstf0, zchi[(kmax - IK + 1) * lmax + (lmax - IL + 1)]);
            sp_amp[(IK - 1) * lmax + (IL - 1)] = a;
        }
    }
    out[0] = alph;
    out[1] = zconstf0;
    out[2] = zgamman;
}

__device__ __forceinline__ uint4 mp_philox(uint4 c, unsigned long long seed) {
    unsigned int k0 = (unsigned int)seed;
    unsigned int k1 = (unsigned int)(seed >> 32);
    for (int round = 0; round < 10; ++round) {
        unsigned long long p0 = 0xD2511F53ULL * c.x;
        unsigned long long p1 = 0xCD9E8D57ULL * c.z;
        c = make_uint4((unsigned int)(p1 >> 32) ^ c.y ^ k0,
                       (unsigned int)p1,
                       (unsigned int)(p0 >> 32) ^ c.w ^ k1,
                       (unsigned int)p0);
        k0 += 0x9E3779B9U;
        k1 += 0xBB67AE85U;
    }
    return c;
}

// gauss_noise (:1488-1514) inside UPDATE_STOCH's |z| < thresh = 3 loop,
// with Philox uniforms: every attempted pair has its own counter.
__device__ float mp_gaussian(unsigned int index, unsigned int step,
                             unsigned int stream, unsigned long long seed) {
    for (unsigned int attempt = 0; ; ++attempt) {
        uint4 r = mp_philox(make_uint4(index, step, stream, attempt), seed);
        const float u = FMUL((float)(r.x >> 8), 0x1p-24f);
        const float v = FMUL((float)(r.y >> 8), 0x1p-24f);
        const float x = FSUB(FMUL(2.0f, u), 1.0f);
        const float y = FSUB(FMUL(2.0f, v), 1.0f);
        const float rr = FADD(FMUL(x, x), FMUL(y, y));
        if (!(rr > 0.0f && rr < 1.0f)) continue;
        const float z = FMUL(FSQRT(FDIV(FMUL(-2.0f, gfk_log(rr)), rr)), x);
        if (fabsf(z) < 3.0f) return z;
    }
}

// ZRANDNOSS and ZRANDNOSC for every (IL, IK) of the domain.
extern "C" __global__ void mp_noise(
    int n, unsigned int step, unsigned int stream_s, unsigned int stream_c,
    unsigned long long seed, float* __restrict__ noss, float* __restrict__ nosc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    noss[i] = mp_gaussian((unsigned int)i, step, stream_s, seed);
    nosc[i] = mp_gaussian((unsigned int)i, step, stream_c, seed);
}

// UPDATE_STOCH's AR(1) step (:908-947) with the draws given.  Each element
// reads only its own old coefficients and the draws: one thread each.
extern "C" __global__ void mp_ar1(
    int lmax, int kmax, float alph, const float* __restrict__ sp_amp,
    const float* __restrict__ noss, const float* __restrict__ nosc,
    float* __restrict__ spforcc, float* __restrict__ spforcs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lmax * kmax) return;
    const int IL = i % lmax + 1, IK = i / lmax + 1;
    const float keep = FSUB(1.f, alph);
    const float amp = sp_amp[i];
#define NS(il, ik) noss[((ik) - 1) * lmax + ((il) - 1)]
#define NC(il, ik) nosc[((ik) - 1) * lmax + ((il) - 1)]
    float c = spforcc[i], s = spforcs[i];
    if (IK <= kmax / 2 + 1 && IK > 1) {
        c = FADD(FMUL(keep, c), FMUL(amp, NC(IL, IK)));
        s = FADD(FMUL(keep, s), FMUL(amp, NS(IL, IK)));
    } else if (IK == 1) {
        if (IL <= lmax / 2 + 1) {
            c = FADD(FMUL(keep, c), FMUL(amp, NC(IL, IK)));
            s = FADD(FMUL(keep, s), FMUL(amp, NS(IL, IK)));
        } else {
            c = FADD(FMUL(keep, c), FMUL(amp, NC(lmax - IL + 2, IK)));
            s = FSUB(FMUL(keep, s), FMUL(amp, NS(lmax - IL + 2, IK)));
        }
    }
    if (IK > kmax / 2 + 1) {
        if (IL <= lmax / 2 + 1 && IL > 1) {
            c = FADD(FMUL(keep, c), FMUL(amp, NC(lmax - IL + 2, kmax - IK + 2)));
            s = FSUB(FMUL(keep, s), FMUL(amp, NS(lmax - IL + 2, kmax - IK + 2)));
        } else if (IL == 1) {
            c = FADD(FMUL(keep, c), FMUL(amp, NC(1, kmax - IK + 2)));
            s = FSUB(FMUL(keep, s), FMUL(amp, NS(1, kmax - IK + 2)));
        } else if (IL > lmax / 2 + 1) {
            c = FADD(FMUL(keep, c), FMUL(amp, NC(lmax - IL + 2, kmax - IK + 2)));
            s = FSUB(FMUL(keep, s), FMUL(amp, NS(lmax - IL + 2, kmax - IK + 2)));
        }
    }
#undef NS
#undef NC
    spforcc[i] = c;
    spforcs[i] = s;
}

// RAND_PERT_UPDATE 'T': RAND_REAL/RAND_IMAG at levels 1..nlev from SPFORCC/S
// (:1188-1196), then for vertstruc = 1 the random-phase rotation (:1222-1233)
// by SETUP_RAND_PERTURB's VERTSTRUCC/S (:816-840, VERTAMP = 1.0), computed
// here where WRF stores them.  WRF's rotation writes RAND_REAL before the
// RAND_IMAG statement reads it; that order is kept.  ``vlevels`` is the
// number of levels SETUP filled (its kts..kte); above it the rotation is by
// the zero VERTSTRUCC/S WRF allocated.  Output [lev][IK][IL].
extern "C" __global__ void mp_fill_levels(
    int lmax, int kmax, int nlev, int vertstruc, int vlevels, float dx, float dy,
    const float* __restrict__ spforcc, const float* __restrict__ spforcs,
    float* __restrict__ rand_real, float* __restrict__ rand_imag)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int plane = lmax * kmax;
    if (i >= plane * nlev) return;
    const int p = i % plane, ilev = i / plane + 1;
    float re = spforcc[p], im = spforcs[p];
    if (vertstruc != 0) {
        const float RPI = 3.141592653589793f;
        const int IL = p % lmax + 1, IK = p / lmax + 1;
        float vc = 0.f, vs = 0.f;
        if (ilev <= vlevels) {
            const float dy2 = FMUL(dy, dy), dx2 = FMUL(dx, dx);
            const float rhoklmax = FSQRT(FADD(FDIV((float)(kmax * kmax), dy2),
                                              FDIV((float)(lmax * lmax), dx2)));
            const float zref = 32.0f;
            const int w = IL <= lmax / 2 ? IL + 1 : lmax - IL + 2;
            const float rhokl = FSQRT(FADD(FDIV((float)((IK + 1) * (IK + 1)), dy2),
                                           FDIV((float)(w * w), dx2)));
            const float eps = FMUL(FMUL(FDIV(FSUB(rhoklmax, rhokl), rhoklmax),
                                        FDIV((float)ilev, zref)), RPI);
            vc = glibc_cosf(FMUL(eps, (float)w));
            vs = glibc_sinf(FMUL(eps, (float)w));
            if (IL > lmax / 2) vs = -vs;
        }
        const float vertamp = 1.0f;
        re = FMUL(vertamp, FSUB(FMUL(re, vc), FMUL(im, vs)));
        im = FMUL(vertamp, FADD(FMUL(re, vs), FMUL(im, vc)));
    }
    rand_real[i] = re;
    rand_imag[i] = im;
}

// RAND_PERT = MAX(MIN(RAND_REAL, thresh), -1.0*thresh) (:1333-1344) on the
// mass points (I < LMAX, J < KMAX): ``pert`` is (nlev, KMAX-1, LMAX-1).
extern "C" __global__ void mp_clip(
    int lmax, int kmax, int nlev, float thresh_fact, float gridpt_stddev,
    const float* __restrict__ rand_real, float* __restrict__ pert)
{
    const int nx = lmax - 1, ny = kmax - 1;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx * ny * nlev) return;
    const int x = i % nx, y = (i / nx) % ny, lev = i / (nx * ny);
    const float thresh = FMUL(thresh_fact, gridpt_stddev);
    const float r = rand_real[(lev * kmax + y) * lmax + x];
    const float lo = FMUL(-1.0f, thresh);
    pert[i] = eps_max(eps_min(r, thresh), lo);
}
