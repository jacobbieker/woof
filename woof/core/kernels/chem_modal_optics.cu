// Clean-room modal aerosol optics for the default build (WOOF P9).
//
// Written from textbook theory only; no WRF-Chem optics source was
// consulted.  For each cell and wavelength, each lognormal mode:
//
// * refractive index of the internal mixture, from its component classes'
//   volumes: the volume average m = sum v_c m_c / sum v_c (mixing = 1), or
//   Maxwell Garnett with elemental carbon as inclusions in the volume-averaged
//   rest (mixing = 2): eps = eps_h (1 + 2 f beta) / (1 - f beta),
//   beta = (eps_i - eps_h) / (eps_i + 2 eps_h), f the carbon volume fraction
//   (Bohren and Huffman 1983, eq. 8.50);
// * number median diameter from the mode's number N and wet volume V,
//   D_g = (6 V / (pi N))^(1/3) exp(-1.5 ln^2 sigma_g) (lognormal third
//   moment, Seinfeld and Pandis 2006 eq. 8.39);
// * the lognormal-integrated Mie optics per unit volume, E_abs, E_sca and g,
//   trilinear in (ln x_g, n, ln k) from the table that
//   tools/chem_modal_optics/make_tables.py builds with WOOF's clean-room Mie
//   code (Bohren and Huffman ch. 4); beta = V E / lambda.  Below the table's
//   smallest x_g the Rayleigh scalings hold (E_abs constant, E_sca ~ x_g^3);
//   above its largest, the geometric limit (E ~ 1/x_g, g constant).
//
// Modes add: tau = (beta_abs + beta_sca) dz, ssa = beta_sca / beta_ext,
// g = sum beta_sca g / sum beta_sca.  The four shortwave wavelengths give
// (tau, ssa, g); the 16 longwave bands give the absorption optical depth
// beta_abs dz, the convention of a no-scattering longwave solver (Pincus et
// al. 2019, JAMES 11, 3074, RTE's absorption-only path).
//
// One thread per cell.

#define MO_NMODE 3
#define MO_NCLASS 6
#define MO_NSW 4
#define MO_NLW 16
#define MO_NWL (MO_NSW + MO_NLW)
#define MO_EC 4                      // elemental carbon's class index

struct MoCx { float re, im; };

__device__ static MoCx mo_mul(MoCx a, MoCx b) {
    MoCx r; r.re = a.re * b.re - a.im * b.im; r.im = a.re * b.im + a.im * b.re; return r;
}
__device__ static MoCx mo_div(MoCx a, MoCx b) {
    const float d = b.re * b.re + b.im * b.im;
    MoCx r; r.re = (a.re * b.re + a.im * b.im) / d; r.im = (a.im * b.re - a.re * b.im) / d; return r;
}
// Principal square root (Im >= 0 for Im(z) >= 0), in the form that avoids
// the cancellation of |z| - Re z when Im z << Re z (a weakly absorbing
// mixture): t = sqrt((|z| + |Re z|) / 2), the other part Im z / (2 t).
__device__ static MoCx mo_sqrt(MoCx z) {
    MoCx r;
    const float t = sqrtf(0.5f * (hypotf(z.re, z.im) + fabsf(z.re)));
    if (!(t > 0.0f)) { r.re = 0.0f; r.im = 0.0f; return r; }
    if (z.re >= 0.0f) {
        r.re = t;
        r.im = z.im / (2.0f * t);
    } else {
        r.re = fabsf(z.im) / (2.0f * t);
        r.im = copysignf(t, z.im);
    }
    return r;
}

// Trilinear lookup of the three quantities (ln E_abs, ln E_sca, g) of one
// sigma; the caller clamps n and ln k and passes ln x_g clamped too.
__device__ static void mo_lookup(const float* __restrict__ table, int s,
                                 float lnx, float n, float lnk,
                                 float lnx0, float dlnx, int nx,
                                 float n0, float dn, int nn,
                                 float lnk0, float dlnk, int nk, float out[3]) {
    float fx = (lnx - lnx0) / dlnx, fn = (n - n0) / dn, fk = (lnk - lnk0) / dlnk;
    int ix = min(max((int)floorf(fx), 0), nx - 2);
    int in = min(max((int)floorf(fn), 0), nn - 2);
    int ik = min(max((int)floorf(fk), 0), nk - 2);
    fx = fminf(fmaxf(fx - (float)ix, 0.0f), 1.0f);
    fn = fminf(fmaxf(fn - (float)in, 0.0f), 1.0f);
    fk = fminf(fmaxf(fk - (float)ik, 0.0f), 1.0f);
    for (int q = 0; q < 3; ++q) {
        const float* t = table + (((long long)(s * 3 + q) * nx) * nn) * nk;
        float acc = 0.0f;
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
                for (int c = 0; c < 2; ++c) {
                    const float w = (a ? fx : 1.0f - fx) * (b ? fn : 1.0f - fn) * (c ? fk : 1.0f - fk);
                    acc += w * t[((long long)(ix + a) * nn + (in + b)) * nk + (ik + c)];
                }
        out[q] = acc;
    }
}

extern "C" __global__ void chem_modal_optics(
    const float* __restrict__ vol,      // (NMODE*NCLASS, ncell) component volume, m3 m-3
    const float* __restrict__ num,      // (NMODE, ncell) number, m-3
    const float* __restrict__ dz,       // (ncell) layer depth, m
    const float* __restrict__ ri_n,     // (NCLASS, NWL)
    const float* __restrict__ ri_k,     // (NCLASS, NWL)
    const float* __restrict__ wl_m,     // (NWL) wavelength, m
    const float* __restrict__ lnsig,    // (NMODE) ln sigma_g
    const float* __restrict__ table,    // (NMODE, 3, nx, nn, nk)
    const float lnx0, const float dlnx, const int nx,
    const float n0, const float dn, const int nn,
    const float lnk0, const float dlnk, const int nk,
    const int mixing,
    float* __restrict__ tau, float* __restrict__ ssa, float* __restrict__ asym,  // (NSW, ncell)
    float* __restrict__ tau_lw,                                                   // (NLW, ncell)
    const long long ncell)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    const float pi = 3.14159265358979f;
    const float lnx1 = lnx0 + dlnx * (float)(nx - 1);
    const float n1 = n0 + dn * (float)(nn - 1), lnk1 = lnk0 + dlnk * (float)(nk - 1);
    float v[MO_NMODE][MO_NCLASS], vtot[MO_NMODE], dg[MO_NMODE];
    for (int m = 0; m < MO_NMODE; ++m) {
        vtot[m] = 0.0f;
        for (int k = 0; k < MO_NCLASS; ++k) {
            v[m][k] = fmaxf(vol[(long long)(m * MO_NCLASS + k) * ncell + c], 0.0f);
            vtot[m] += v[m][k];
        }
        const float n = num[(long long)m * ncell + c];
        dg[m] = (vtot[m] > 0.0f && n > 0.0f)
                ? powf(6.0f * vtot[m] / (pi * n), 1.0f / 3.0f) * expf(-1.5f * lnsig[m] * lnsig[m])
                : 0.0f;
    }
    const float h = dz[c];
    for (int l = 0; l < MO_NWL; ++l) {
        float babs = 0.0f, bsca = 0.0f, bg = 0.0f;
        for (int m = 0; m < MO_NMODE; ++m) {
            if (!(dg[m] > 0.0f)) continue;
            MoCx mix;
            if (mixing == 2 && v[m][MO_EC] > 0.0f && vtot[m] > v[m][MO_EC]) {
                MoCx host; host.re = 0.0f; host.im = 0.0f;
                const float vh = vtot[m] - v[m][MO_EC];
                for (int k = 0; k < MO_NCLASS; ++k) {
                    if (k == MO_EC) continue;
                    host.re += v[m][k] * ri_n[k * MO_NWL + l];
                    host.im += v[m][k] * ri_k[k * MO_NWL + l];
                }
                host.re /= vh; host.im /= vh;
                MoCx inc; inc.re = ri_n[MO_EC * MO_NWL + l]; inc.im = ri_k[MO_EC * MO_NWL + l];
                const MoCx eh = mo_mul(host, host), ei = mo_mul(inc, inc);
                MoCx num_, den;
                num_.re = ei.re - eh.re; num_.im = ei.im - eh.im;
                den.re = ei.re + 2.0f * eh.re; den.im = ei.im + 2.0f * eh.im;
                const MoCx beta = mo_div(num_, den);
                const float f = v[m][MO_EC] / vtot[m];
                MoCx top, bot;
                top.re = 1.0f + 2.0f * f * beta.re; top.im = 2.0f * f * beta.im;
                bot.re = 1.0f - f * beta.re; bot.im = -f * beta.im;
                mix = mo_sqrt(mo_mul(eh, mo_div(top, bot)));
            } else {
                mix.re = 0.0f; mix.im = 0.0f;
                for (int k = 0; k < MO_NCLASS; ++k) {
                    mix.re += v[m][k] * ri_n[k * MO_NWL + l];
                    mix.im += v[m][k] * ri_k[k * MO_NWL + l];
                }
                mix.re /= vtot[m]; mix.im /= vtot[m];
            }
            const float lam = wl_m[l];
            const float lnx = logf(pi * dg[m] / lam);
            const float nre = fminf(fmaxf(mix.re, n0), n1);
            const float lnk = fminf(fmaxf(logf(fmaxf(mix.im, 1.0e-30f)), lnk0), lnk1);
            const float lnxc = fminf(fmaxf(lnx, lnx0), lnx1);
            float q[3];
            mo_lookup(table, m, lnxc, nre, lnk, lnx0, dlnx, nx, n0, dn, nn, lnk0, dlnk, nk, q);
            float lea = q[0], les = q[1];
            if (lnx < lnx0) {
                les += 3.0f * (lnx - lnx0);           // Rayleigh: E_sca ~ x^3
            } else if (lnx > lnx1) {
                lea -= lnx - lnx1;                    // geometric: E ~ 1/x
                les -= lnx - lnx1;
            }
            const float scale = vtot[m] / lam;
            const float ba = scale * expf(lea), bs = scale * expf(les);
            babs += ba;
            bsca += bs;
            bg += bs * fminf(fmaxf(q[2], -1.0f), 1.0f);
        }
        if (l < MO_NSW) {
            const float bext = babs + bsca;
            tau[(long long)l * ncell + c] = bext * h;
            ssa[(long long)l * ncell + c] = bext > 0.0f ? bsca / bext : 1.0f;
            asym[(long long)l * ncell + c] = bsca > 0.0f ? bg / bsca : 0.0f;
        } else {
            tau_lw[(long long)(l - MO_NSW) * ncell + c] = babs * h;
        }
    }
}
