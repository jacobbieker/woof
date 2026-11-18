// Climatological ionosphere on model columns (space-weather layer L3a).
//
// An IRI-type profile from published formulas, one thread per column.  It was
// written from the papers below, with PyIRI (Forsythe et al. 2023, MIT
// licence) read for the arrangement of the CCIR/URSI expansions; no IRI or
// PyIRI code is executed or copied.
//
// * foF2 and M(3000)F2: Jones & Gallet 1965 (ITU Telecomm. J. 32, 18) and
//   Jones & Graham 1966 numerical maps, CCIR Report 340 / URSI 1988
//   coefficients.  The host contracts the coefficients with the diurnal
//   Fourier series (UT) and blends months and solar activity (foF2 in IG12,
//   M(3000)F2 in R12, linear between 0 and 100, CCIR Report 340 convention);
//   here only the geographic functions G_k(modip, lat, lon) remain:
//       G = sin(modip)^i cos(lat)^j {cos, sin}(j lon),  i < Q[j]
//   (foF2: Q = 12 12 9 5 2 1 1 1 1, 76 terms; M3000: Q = 7 8 6 3 2 1 1, 49).
// * foE: Kouris & Muggleton 1973 with the IRI effective zenith angle
//   (Bilitza 1990, NSSDC 90-22; Bilitza et al. 2022, Rev. Geophys. 60):
//       chi_eff = (f1 e + chi) / (e + 1),  e = exp(12 (chi - 86.233)),
//       f1 = 90 - 0.24 exp(20 - 0.2 chi)
//       foE^2 = 0.49 + (1.112 - 0.019 s)^2 sqrt(F10.7) cos(chi_eff)^0.6
// * hmF2: Bilitza, Sheikh & Eyfrig 1979 (Telecomm. J. 46, 549):
//       hmF2 = 1490 / (M3000 + dM) - 176,
//       dM = f1 f2 / (foF2/foE - f3) + f4, foF2/foE >= 1.7.
// * F2 bottomside thickness: Mosert de Gonzalez & Radicella 1990
//   (dN/dh at the peak) and the topside thickness of Bilitza 1990
//   (k = 3.22 - 0.0538 foF2 - 0.00664 hmF2 + 0.113 hmF2/B + 0.00257 R12).
// * F1: Ducharme et al. 1971 and Scotto et al. 1997 (occurrence), with the
//   Epstein-layer profile of Rawer 1988; E-F transition by the 4th-power
//   drop functions; topside Epstein with the NeQuick-type height scaling
//   (g = 0.125, r = 100; Radicella & Leitinger 2001, Adv. Space Res. 27, 35).
//
// Densities are m-3 (Nm = 1.24e10 fo^2, fo in MHz).  Vertical TEC (TECU)
// integrates the profile between alt_lo and alt_hi (km) with nalt midpoint
// cells.  float throughout; the kernel is shared by CUDA and Metal.

#define IONO_NK_F2 76
#define IONO_NK_M3 49
#define IONO_DEG2RAD 0.017453292519943295f
#define IONO_CHI0 86.23292796211615f

__device__ static float iono_fexp(float x) {
    if (x > 80.0f) return 5.5406e34f;
    if (x < -80.0f) return 1.8049e-35f;
    return expf(x);
}

// Epstein layer, peak Nm at hm, scale B; zero above alpha > 25 as in IRI.
__device__ static float iono_epstein(float nm, float hm, float b, float h) {
    const float alpha = (b != 0.0f) ? (h - hm) / b : 30.0f;
    if (alpha > 25.0f) return 0.0f;
    const float e = iono_fexp(alpha);
    const float d = 1.0f + e;
    return 4.0f * nm * e / (d * d);
}

__device__ static float iono_epstein_top(float nm, float hm, float b, float h) {
    const float g = 0.125f;
    const float r = 100.0f;
    const float dh = h - hm;
    const float z = dh / (b * (1.0f + r * g * dh / (r * b + g * dh)));
    const float e = iono_fexp(z);
    if (e > 1.0e11f) return 4.0f * nm / e;
    const float d = e + 1.0f;
    return 4.0f * nm * e / (d * d);
}

// sum_k c_k G_k for one CCIR expansion (q: the highest sine power per order).
__device__ static float iono_expand(const float *c, const int *q, int nq,
                                    float smu, float clat, float lon) {
    float acc = 0.0f;
    int k = 0;
    float cj = 1.0f;  // cos(lat)^j
    for (int j = 0; j < nq; ++j) {
        const float cl = cosf((float)j * lon);
        const float sl = sinf((float)j * lon);
        float si = 1.0f;  // sin(modip)^i
        for (int i = 0; i < q[j]; ++i) {
            const float base = si * cj;
            if (j == 0) {
                acc += c[k] * base;
                k += 1;
            } else {
                acc += c[k] * base * cl;
                acc += c[k + 1] * base * sl;
                k += 2;
            }
            si *= smu;
        }
        cj *= clat;
    }
    return acc;
}

__device__ static float iono_density(float h, float nmf2, float hmf2, float bf2bot, float bf2top,
                                     int has_f1, float nmf1, float hmf1, float bf1bot,
                                     float nme, float hme) {
    const float be_bot = 5.0f;
    const float be_top = 7.0f;
    float ne = 0.0f;
    if (h >= hmf2) {
        ne = iono_epstein_top(nmf2, hmf2, bf2top, h);
    } else if (h <= hme) {
        ne = iono_epstein(nme, hme, be_bot, h);
    } else if (has_f1) {
        if (h >= hmf1) {
            ne = iono_epstein(nmf2, hmf2, bf2bot, h);
        } else {
            const float span = hmf1 - hme;
            const float u1 = (h - hme) / span;
            const float u2 = (hmf1 - h) / span;
            const float d1 = 1.0f - u1 * u1 * u1 * u1;
            const float d2 = 1.0f - u2 * u2 * u2 * u2;
            ne = iono_epstein(nme, hme, be_top, h) * d1
               + iono_epstein(nmf1, hmf1, bf1bot, h) * d2;
        }
    } else {
        const float span = hmf2 - hme;
        const float u1 = (h - hme) / span;
        const float u2 = (hmf2 - h) / span;
        const float d1 = 1.0f - u1 * u1 * u1 * u1;
        const float d2 = 1.0f - u2 * u2 * u2 * u2;
        ne = iono_epstein(nme, hme, be_top, h) * d1
           + iono_epstein(nmf2, hmf2, bf2bot, h) * d2;
    }
    return ne > 1.0f ? ne : 1.0f;
}

extern "C" __global__ void spacewx_ionosphere(
    const float *lat_deg, const float *lon_deg, const float *modip_deg, const float *diplat_deg,
    const float *coef,        // 76 foF2 then 49 M(3000)F2 contracted coefficients
    const float *fo_scale,    // per column factor on foF2 (1: the climatology; woof.spacewx.storm)
    const float r12, const float f107, const float sun_lat_deg, const float sun_lon_deg,
    const int season, const float alt_lo, const float alt_hi, const int nalt,
    float *fof2, float *m3000, float *hmf2_out, float *nmf2_out, float *foe_out,
    float *fof1_out, float *hmf1_out, float *bbot_out, float *btop_out, float *tec,
    const long long ncol)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const int q_f2[9] = {12, 12, 9, 5, 2, 1, 1, 1, 1};
    const int q_m3[7] = {7, 8, 6, 3, 2, 1, 1};
    const float lat = lat_deg[c] * IONO_DEG2RAD;
    const float lon = lon_deg[c] * IONO_DEG2RAD;
    const float smu = sinf(modip_deg[c] * IONO_DEG2RAD);
    const float clat = cosf(lat);

    // F2 peak (CCIR/URSI maps).
    float fo = iono_expand(coef, q_f2, 9, smu, clat, lon);
    const float m3 = iono_expand(coef + IONO_NK_F2, q_m3, 7, smu, clat, lon);
    fo = (fo > 0.0f ? fo : 0.0f) * fo_scale[c];

    // Solar zenith angle from the subsolar point.
    const float slat = sun_lat_deg * IONO_DEG2RAD;
    float cchi = sinf(slat) * sinf(lat)
               + cosf(slat) * cosf(lat) * cosf(sun_lon_deg * IONO_DEG2RAD - lon);
    cchi = fminf(1.0f, fmaxf(-1.0f, cchi));
    const float chi = acosf(cchi) / IONO_DEG2RAD;

    // E peak.
    const float ee = iono_fexp(12.0f * (chi - IONO_CHI0));
    const float f1c = 90.0f - 0.24f * iono_fexp(20.0f - 0.2f * chi);
    const float chi_eff = (f1c * ee + chi) / (ee + 1.0f);
    const float es = iono_fexp(0.3f * lat_deg[c] * IONO_DEG2RAD);
    const float seasp = (float)season * (es - 1.0f) / (es + 1.0f);
    const float ce = fmaxf(0.0f, cosf(chi_eff * IONO_DEG2RAD));
    const float sfac = 1.112f - 0.019f * seasp;
    const float foe = sqrtf(0.49f + sfac * sfac * sqrtf(f107) * powf(ce, 0.6f));

    // hmF2 (Bilitza et al. 1979).
    const float mdp = modip_deg[c] / 40.0f;
    const float g1 = 0.00232f * r12 + 0.222f;
    const float g2 = 1.0f - r12 / 150.0f * expf(-mdp * mdp);
    const float g3 = 1.2f - 0.0116f * expf(r12 / 41.84f);
    const float g4 = 0.096f * (r12 - 25.0f) / 150.0f;
    float ratio = foe > 0.0f ? fo / foe : 1.7f;
    ratio = ratio < 1.7f ? 1.7f : ratio;
    const float dm = g1 * g2 / (ratio - g3) + g4;
    const float hmf2 = 1490.0f / (m3 + dm) - 176.0f;
    const float hme = 110.0f;

    // Densities and F2 thicknesses.
    const float nmf2 = fmaxf(1.24e10f * fo * fo, 1.0e6f);
    const float nme = fmaxf(1.24e10f * foe * foe, 1.0e6f);
    const float dndh = 0.01f * iono_fexp(-3.467f + 1.714f * logf(fmaxf(fo, 1.0e-3f))
                                         + 2.02f * logf(fmaxf(m3, 1.0e-3f)));
    const float bbot = 3.85e-12f * nmf2 / dndh;
    const float k = 3.22f - 0.0538f * fo - 0.00664f * hmf2 + 0.113f * hmf2 / bbot + 0.00257f * r12;
    const float x = (k * bbot - 150.0f) / 100.0f;
    float btop = (100.0f * x + 150.0f) / (0.041163f * x * x - 0.183981f * x + 1.424472f);
    btop = btop > 0.0f ? btop : 30.0f;

    // F1 layer where the Sun is high enough (multiplier >= 0).
    const float mdl = fabsf(diplat_deg[c]);
    const float n_exp = 0.093f + 0.0046f * mdl - 0.000054f * mdl * mdl + 0.0003f * r12;
    const float f100 = 5.348f + 0.011f * mdl - 0.00023f * mdl * mdl;
    const float f0 = 4.35f + 0.0058f * mdl - 0.00012f * mdl * mdl;
    const float fs = f0 + (f100 - f0) * r12 / 100.0f;
    const float raw_f1 = cchi > 0.0f ? fs * powf(cchi, n_exp) : 0.0f;
    float mult = fminf(-10.0f + 30.0f * cchi, 10.0f) / 10.0f;
    int has_f1 = mult >= 0.0f ? 1 : 0;
    float fof1 = 0.0f, hmf1 = 0.0f, nmf1 = 1.0f, bf1 = 10.0f;
    if (has_f1) {
        fof1 = mult * raw_f1 + (1.0f - mult) * foe;
        nmf1 = 1.24e10f * fof1 * fof1;
        nmf1 = nmf1 > 0.0f ? nmf1 : 1.0f;
        const float b = 2.0f - 4.0f * nmf2 / nmf1;
        const float d = b * b - 4.0f;
        float root = 0.0f;
        if (d >= 0.0f) root = 0.5f * (-b - sqrtf(d));
        hmf1 = root > 0.0f ? bbot * logf(root) + hmf2 : 0.0f;
        if (hmf1 <= 110.0f) {
            has_f1 = 0;
        } else {
            bf1 = 0.5f * (hmf1 - hme);
            bf1 = bf1 > 0.0f ? bf1 : 10.0f;
        }
    }

    // Vertical TEC by midpoint cells.
    const float dz = (alt_hi - alt_lo) / (float)nalt;
    float acc = 0.0f;
    for (int i = 0; i < nalt; ++i) {
        const float h = alt_lo + ((float)i + 0.5f) * dz;
        acc += iono_density(h, nmf2, hmf2, bbot, btop, has_f1, nmf1, hmf1, bf1, nme, hme);
    }

    fof2[c] = fo;
    m3000[c] = m3;
    hmf2_out[c] = hmf2;
    nmf2_out[c] = nmf2;
    foe_out[c] = foe;
    fof1_out[c] = has_f1 ? fof1 : 0.0f;
    hmf1_out[c] = has_f1 ? hmf1 : 0.0f;
    bbot_out[c] = bbot;
    btop_out[c] = btop;
    tec[c] = acc * dz * 1.0e-13f;   // km -> m (1e3) and m-2 -> TECU (1e-16)
}
