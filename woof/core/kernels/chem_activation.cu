// Clean-room multimode aerosol activation for the default build (WOOF P8).
//
// Written from the published papers only; no WRF-Chem source was consulted.
//
// * Abdul-Razzak, Ghan and Rivera-Carpio 1998, JGR 103(D6), 6123-6131
//   (ARG98): the growth coefficient G (eq. 16), alpha (eq. 11), gamma
//   (eq. 12), the Kelvin coefficient A (eq. 5) and the critical
//   supersaturation of a mode's median radius (eq. 9).
// * Abdul-Razzak and Ghan 2000, JGR 105(D5), 6837-6844 (ARG00): the
//   multimode maximum supersaturation (eqs. 6, 7-10: f_i, g_i, zeta, eta_i)
//   and the activated number (eq. 13) and mass (eq. 14) fractions of a
//   lognormal mode.  The surface fraction is the same erfc at the surface
//   median radius, ln r_s = ln r_g + 2 ln^2(sigma) (Seinfeld and Pandis,
//   2006, eq. 8.39 for the lognormal moments).
// * Ghan et al. 1997, JGR 102(D18), 21777-21794: the droplet nucleation
//   source over a Gaussian sub-grid updraft distribution,
//       fn    = int_{w>0} f(w) p(w) dw,     fluxn = int_{w>0} w f(w) p(w) dw,
//       flux_fullact = int_{w>0} w p(w) dw,
//   evaluated here by composite Simpson over the updraft interval
//   [max(wmin, 0, wbar - 8 sigw), min(wmax, wbar + 8 sigw)].
// * Thermodynamics: Bolton 1980 (MWR 108, 1046) saturation vapour pressure;
//   Pruppacher and Klett 1997, eqs. 13.3 (vapour diffusivity) and 13.18a
//   (thermal conductivity of air), continuum values.
//
// One thread per case.  Inputs per mode are (nmode, ncase) planes: number
// na (m-3), volume volc (m3 m-3 dry), geometric standard deviation sigman
// and the ARG hygroscopicity B (hygro).  Per case: mean updraft wbar, its
// standard deviation sigw (<= SIGW_DELTA: a single updraft), a diabatic
// cooling expressed as an updraft wdiab (adds to w in the supersaturation
// only), the updraft bounds wmin/wmax (m s-1), air temperature (K) and
// density (kg m-3), and a prescribed maximum supersaturation (negative:
// none, use ARG00).  Outputs: fn, fs, fm, fluxn, fluxs, fluxm per mode,
// flux_fullact and the maximum supersaturation at wbar + wdiab per case.

#define ACT_MAXMODE 8
#define ACT_NQ 64              // Simpson intervals over the updraft range
#define SIGW_DELTA 1.0e-4f     // m s-1

// Physical constants (SI).
#define MW_WATER 0.018015f     // kg mol-1
#define MW_AIR 0.028966f       // kg mol-1
#define R_UNIV 8.314f          // J mol-1 K-1
#define R_DRY 287.04f          // J kg-1 K-1
#define RHO_WATER 1000.0f      // kg m-3
#define L_VAP 2.501e6f         // J kg-1, latent heat of vaporisation at 0 C
#define CP_AIR 1004.6f         // J kg-1 K-1
#define GRAV 9.81f             // m s-2
#define SURF_TENSION 0.076f    // N m-1, water/air (ARG98 sec. 2)
#define PI_F 3.14159265358979f

struct ActThermo {
    float A;       // ARG98 eq. 5, m
    float aw_g;    // alpha / G (multiplies w), ARG98 eqs. 11 and 16
    float gamma;   // ARG98 eq. 12
};

__device__ static ActThermo act_thermo(float T, float rho) {
    ActThermo th;
    const float p = rho * R_DRY * T;
    const float es = 611.2f * expf(17.67f * (T - 273.15f) / (T - 29.65f));
    const float dv = 0.211e-4f * powf(T / 273.15f, 1.94f) * (101325.0f / p);
    const float ka = 1.0e-3f * (4.39f + 0.071f * T);
    th.A = 2.0f * SURF_TENSION * MW_WATER / (RHO_WATER * R_UNIV * T);
    const float alpha = GRAV * MW_WATER * L_VAP / (CP_AIR * R_UNIV * T * T)
                        - GRAV * MW_AIR / (R_UNIV * T);
    th.gamma = R_UNIV * T / (es * MW_WATER)
               + MW_WATER * L_VAP * L_VAP / (CP_AIR * p * MW_AIR * T);
    const float ginv = RHO_WATER * R_UNIV * T / (es * dv * MW_WATER)
                       + L_VAP * RHO_WATER / (ka * T) * (L_VAP * MW_WATER / (R_UNIV * T) - 1.0f);
    th.aw_g = alpha * ginv;
    return th;
}

// ARG00 eq. 6: maximum supersaturation for updraft w (> 0).
__device__ static float act_smax(const ActThermo& th, float w, int nmode,
                                 const float* n, const float* sm, const float* lnsg) {
    const float awg = th.aw_g * w;
    const float zeta = 2.0f * th.A / 3.0f * sqrtf(awg);
    const float awg32 = awg * sqrtf(awg);
    float sum = 0.0f;
    for (int m = 0; m < nmode; ++m) {
        if (!(n[m] > 0.0f) || !(sm[m] > 0.0f)) continue;
        const float eta = awg32 / (2.0f * PI_F * RHO_WATER * th.gamma * n[m]);
        const float f = 0.5f * expf(2.5f * lnsg[m] * lnsg[m]);
        const float g = 1.0f + 0.25f * lnsg[m];
        const float ze = zeta / eta;
        const float s2 = sm[m] * sm[m];
        sum += (f * ze * sqrtf(ze) + g * powf(s2 / (eta + 3.0f * zeta), 0.75f)) / s2;
    }
    return sum > 0.0f ? 1.0f / sqrtf(sum) : 0.0f;
}

// Activated number, surface and mass fractions of every mode at one
// updraft (ARG00 eqs. 13-14), into out[3*m .. 3*m+2].
__device__ static void act_fractions(const ActThermo& th, float weff, float sgiven,
                                     int nmode, const float* n, const float* sm,
                                     const float* lnsg, float* out) {
    const float sqrt2 = 1.41421356237f;
    const float smx = sgiven >= 0.0f ? sgiven
                      : (weff > 0.0f ? act_smax(th, weff, nmode, n, sm, lnsg) : 0.0f);
    for (int m = 0; m < nmode; ++m) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
        if (sm[m] > 0.0f && smx > 0.0f) {
            const float u = 2.0f * logf(sm[m] / smx) / (3.0f * sqrt2 * lnsg[m]);
            a0 = 0.5f * erfcf(u);                          // number, ARG00 eq. 13
            a1 = 0.5f * erfcf(u - sqrt2 * lnsg[m]);         // surface
            a2 = 0.5f * erfcf(u - 1.5f * sqrt2 * lnsg[m]);  // mass, ARG00 eq. 14
        }
        out[3 * m] = a0;
        out[3 * m + 1] = a1;
        out[3 * m + 2] = a2;
    }
}

extern "C" __global__ void chem_activation_modal(
    const float* __restrict__ na, const float* __restrict__ volc,
    const float* __restrict__ sigman, const float* __restrict__ hygro,
    const float* __restrict__ wbar, const float* __restrict__ sigw,
    const float* __restrict__ wdiab, const float* __restrict__ wmin,
    const float* __restrict__ wmax, const float* __restrict__ tair,
    const float* __restrict__ rhoair, const float* __restrict__ smax_given,
    float* __restrict__ fn, float* __restrict__ fs, float* __restrict__ fm,
    float* __restrict__ fluxn, float* __restrict__ fluxs, float* __restrict__ fluxm,
    float* __restrict__ flux_fullact, float* __restrict__ smax_out,
    const int nmode, const long long ncase)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncase) return;
    float n[ACT_MAXMODE], sm[ACT_MAXMODE], lnsg[ACT_MAXMODE];
    float f[3 * ACT_MAXMODE], acc[3 * ACT_MAXMODE], fl[3 * ACT_MAXMODE];
    const ActThermo th = act_thermo(tair[c], rhoair[c]);
    for (int m = 0; m < nmode; ++m) {
        const long long i = (long long)m * ncase + c;
        n[m] = na[i];
        lnsg[m] = logf(fmaxf(sigman[i], 1.0001f));
        sm[m] = 0.0f;
        if (n[m] > 0.0f && volc[i] > 0.0f && hygro[i] > 0.0f) {
            // Number median radius from the mode's number and volume
            // (lognormal third moment: V = N pi/6 Dg^3 exp(4.5 ln^2 sg)).
            const float rg = 0.5f * powf(6.0f * volc[i] / (PI_F * n[m]), 1.0f / 3.0f)
                             * expf(-1.5f * lnsg[m] * lnsg[m]);
            const float t = th.A / (3.0f * rg);
            sm[m] = 2.0f / sqrtf(hygro[i]) * t * sqrtf(t);   // ARG98 eq. 9
        } else {
            n[m] = 0.0f;
        }
    }
    for (int k = 0; k < 3 * nmode; ++k) { acc[k] = 0.0f; fl[k] = 0.0f; }
    const float wd = wdiab[c], sgiven = smax_given[c];
    const float wb = wbar[c], sw = sigw[c];
    float fullact = 0.0f;
    if (sw <= SIGW_DELTA) {
        // A single updraft wbar (Ghan et al. 1997 with a delta distribution).
        if (wb > 0.0f) {
            act_fractions(th, wb + wd, sgiven, nmode, n, sm, lnsg, f);
            for (int k = 0; k < 3 * nmode; ++k) { acc[k] = f[k]; fl[k] = wb * f[k]; }
            fullact = wb;
        }
    } else {
        const float lo = fmaxf(fmaxf(wmin[c], 0.0f), wb - 8.0f * sw);
        const float hi = fminf(wmax[c], wb + 8.0f * sw);
        if (hi > lo) {
            const float h = (hi - lo) / (float)ACT_NQ;
            const float norm = 1.0f / (sqrtf(2.0f * PI_F) * sw);
            for (int j = 0; j <= ACT_NQ; ++j) {
                const float w = lo + h * (float)j;
                const float z = (w - wb) / sw;
                const float wt = (j == 0 || j == ACT_NQ) ? 1.0f : ((j & 1) ? 4.0f : 2.0f);
                const float q = wt * h / 3.0f * norm * expf(-0.5f * z * z);
                fullact += q * w;
                if (!(w > 0.0f)) continue;
                act_fractions(th, w + wd, sgiven, nmode, n, sm, lnsg, f);
                for (int k = 0; k < 3 * nmode; ++k) {
                    acc[k] += q * f[k];
                    fl[k] += q * w * f[k];
                }
            }
        }
    }
    for (int m = 0; m < nmode; ++m) {
        const long long i = (long long)m * ncase + c;
        fn[i] = acc[3 * m]; fs[i] = acc[3 * m + 1]; fm[i] = acc[3 * m + 2];
        fluxn[i] = fl[3 * m]; fluxs[i] = fl[3 * m + 1]; fluxm[i] = fl[3 * m + 2];
    }
    flux_fullact[c] = fullact;
    smax_out[c] = sgiven >= 0.0f ? sgiven
                  : (wb + wd > 0.0f ? act_smax(th, wb + wd, nmode, n, sm, lnsg) : 0.0f);
}
