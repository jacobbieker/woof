// WRF v4.7.1 aerosol-aware Morrison (mp_physics = 40): the prescribed-
// aerosol procedures of phys/module_mp_morr_two_moment_aero.F that the
// mp_physics = 10 transcription (morrison.cu) does not have, transcribed
// from the oracle build's -O0 GIMPLE (tools/morrison_aero_wrf471_oracle):
// DERF1 :5334-5445, mdm_prescribed_activate :5628-5768 (Abdul-Razzak and
// Ghan 2000), mdm_prescribed_maxsat :5770-5815, mdm_prescribed_nucleati
// :5817-5972 (Liu and Penner 2005 with Meyers 1992), mdm_prescribed_hetero
// :5974-6007 and mdm_prescribed_hf :6009-6081.  The words their init
// computes are morrison_aero_constants.cuh's.  Same discipline as
// morrison.cu: REAL operations in float in gfortran's order, every
// division __fdiv_rn, EXP/LOG/** WOOF's own glibc words (gfk_exp, gfk_log,
// gfk_pow), SQRT the IEEE square root, and Fortran MIN(a, b) of two
// variables a < b ? a : b (the convention measured for morrison.cu).
//
// The procedures keep WRF's argument order; the ten modes are WRF's
// aername order (sulfate, seasalt2, dust1-4, ocpho, bcpho, ocphi, bcphi).

#define MA_NMODE 10

__device__ __forceinline__ float ma_sqrt(float x) { return __fsqrt_rn(x); }

// DERF1 (the module's own error function).
__device__ float ma_derf1(float x)
{
    const float w = fabsf(x);
    float y;
    if ((double)w < 2.2) {
        float t = w * w;
        int k = (int)t;
        t = t - (float)k;
        k = k * 13;
        const float* a = ma_derf1_a + k;
        y = ((((((((((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t
                    + a[4]) * t + a[5]) * t + a[6]) * t + a[7]) * t
                + a[8]) * t + a[9]) * t + a[10]) * t + a[11]) * t
             + a[12]) * w;
    } else if ((double)w < 6.9) {
        int k = (int)w;
        const float t = w - (float)k;
        k = (k - 2) * 13;
        const float* b = ma_derf1_b + k;
        y = (((((((((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t
                   + b[4]) * t + b[5]) * t + b[6]) * t + b[7]) * t
               + b[8]) * t + b[9]) * t + b[10]) * t + b[11]) * t + b[12];
        y = y * y;
        y = y * y;
        y = y * y;
        y = 1.0f - y * y;
    } else {
        y = 1.0f;
    }
    if (x < 0.0f) y = -y;
    return y;
}

// mdm_prescribed_maxsat.  Every call sets SMAX (a weak mode sets 1e-20
// before the next mode is tested; the first strong mode computes all).
__device__ float ma_maxsat(float zeta, const float* eta, const float* smc)
{
    float smax = 0.0f;
    bool strong = false;
    for (int m = 0; m < MA_NMODE; ++m) {
        if (zeta > eta[m] * 1.0e5f || smc[m] * smc[m] > eta[m] * 1.0e5f) {
            smax = 1.0e-20f;
        } else {
            strong = true;
            break;
        }
    }
    if (!strong) return smax;
    float sum = 0.0f;
    for (int m = 0; m < MA_NMODE; ++m) {
        if (eta[m] > 1.0e-20f) {
            float g1 = ma_sqrt(FDIV(zeta, eta[m]));
            g1 = (g1 * g1) * g1;
            float g2 = FDIV(smc[m], ma_sqrt(eta[m] + zeta * 3.0f));
            g2 = ma_sqrt(g2);
            g2 = (g2 * g2) * g2;
            sum = FDIV(ma_f1_pamdm[m] * g1 + ma_f2_pamdm[m] * g2,
                       smc[m] * smc[m]) + sum;
        } else {
            sum = 1.0e20f;
        }
    }
    return FDIV(1.0f, ma_sqrt(sum));
}

// mdm_prescribed_activate: the number activated (kg-1) by updraft wbar.
__device__ float ma_activate(float wbar, float tair, float rhoair,
                             const float* naero, const float* maero,
                             float latvap)
{
    if (wbar <= 0.0f) return 0.0f;
    const float pres = (rhoair * 2.87e2f) * tair;
    const float diff0 = (FDIV(1.01325e5f, pres) * 2.11e-5f)
        * gfk_pow(FDIV(tair, 300.0f), 1.94f);
    const float conduct0 = (((tair - 300.0f) * 0.017f + 5.69f) * 418.6f)
        * 1.0e-5f;
    // 1000.*svp1 folded by gfortran.
    const float es = gfk_exp(FDIV((tair - 300.0f) * 17.67f, tair - 29.65f))
        * 6.11199951171875e2f;
    const float qs = FDIV(es * MO_EP_2, pres - es);
    const float dqsdt = FDIV(latvap, (tair * MO_RV) * tair) * qs;
    // CP*RV folded by gfortran.
    const float alpha = (FDIV(latvap, (tair * 4.6367721875e5f) * tair)
                         - FDIV(1.0f, tair * 2.87e2f)) * MO_G;
    const float gammaloc = FDIV(FDIV(latvap, MO_CP) * dqsdt + 1.0f,
                                rhoair * qs);
    const float gloc = FDIV(1.0f,
        FDIV(MO_RHOW, (rhoair * diff0) * qs)
        + FDIV(latvap * MO_RHOW, tair * conduct0)
          * (FDIV(latvap, tair * MO_RV) - 1.0f));
    const float sqrtg = ma_sqrt(gloc);
    const float beta = ((MO_RHOW * 1.256637096405029296875e1f) * gloc)
        * gammaloc;
    const float etafactor2max = FDIV(1.0e10f, gfk_pow(wbar * alpha, 1.5f));
    float smc[MA_NMODE], eta[MA_NMODE], lnsm[MA_NMODE];
    for (int m = 0; m < MA_NMODE; ++m) {
        const float volc = FDIV(maero[m], ma_density_aer[m]);
        if (volc > MA_REAL_TINY && naero[m] > MA_REAL_TINY) {
            eta[m] = FDIV(1.0f, (naero[m] * beta) * sqrtg);
            const float amcube = FDIV(volc * 3.0f,
                (ma_exp45logsig_pamdm[m] * 1.256637096405029296875e1f)
                * naero[m]);
            if (ma_hygro_aer[m] > 1.0e-10f) {
                smc[m] = (MA_ATEN_PAMDM * 2.0f) * ma_sqrt(FDIV(MA_ATEN_PAMDM,
                    (ma_hygro_aer[m] * 27.0f) * amcube));
            } else {
                smc[m] = 100.0f;
            }
        } else {
            smc[m] = 1.0f;
            eta[m] = etafactor2max;
        }
        lnsm[m] = gfk_log(smc[m]);
    }
    const float alw = alpha * wbar;
    const float sqrtalw = ma_sqrt(alw);
    const float zeta = FDIV((sqrtalw * 2.0f) * MA_ATEN_PAMDM, sqrtg * 3.0f);
    const float etafactor1 = (alw * 2.0f) * sqrtalw;
    for (int m = 0; m < MA_NMODE; ++m) eta[m] = eta[m] * etafactor1;
    const float lnsmax = gfk_log(ma_maxsat(zeta, eta, smc));
    float nact = 0.0f;
    for (int m = 0; m < MA_NMODE; ++m) {
        const float x = FDIV((lnsm[m] - lnsmax) * 2.0f,
                             (MA_SQ2_PAMDM * 3.0f) * ma_alogsig_pamdm[m]);
        nact = nact + ((1.0f - ma_derf1(x)) * 0.5f) * naero[m];
    }
    return FDIV(nact, rhoair);
}

// mdm_prescribed_hetero: immersion nucleation (cm-3); Nid is always 0.
__device__ float ma_hetero(float t, float ww, float ns)
{
    const float a11 = 0.0263f, a12 = -0.0185f, a21 = 2.758f, a22 = 1.3221f;
    const float b11 = -0.008f, b12 = -0.0468f, b21 = -0.2667f;
    const float b22 = -1.4588f;
    const float b = (gfk_log(ns) * b11 + a11) * gfk_log(ww)
        + (gfk_log(ns) * b12 + a12);
    const float c = gfk_log(ns) * b21 + a21;
    const float nis = ((gfk_exp(a22) * gfk_pow(ns, b22)) * gfk_exp(t * b))
        * gfk_pow(ww, c);
    return nis < ns ? nis : ns;
}

// mdm_prescribed_hf: homogeneous freezing of sulfate (cm-3).
__device__ float ma_hf(float t, float ww, float rh, float subgrid, float na)
{
    float ni = 0.0f;
    const float a = gfk_log(ww) * 6.0e-4f + 6.6e-3f;
    const float b = gfk_log(ww) * 6.0e-2f + 1.052f;
    const float c = gfk_log(ww) * 1.68f + 129.35f;
    const float rhw = (((t * a) * t + t * b) + c) * 0.01f;
    if (t <= -37.0f && rh * subgrid >= rhw) {
        const float regm = gfk_log(ww) * 6.07f - 55.0f;
        if (t >= regm) {
            const float a2 = t > -64.0f ? -1.6387f : -6.045f;
            const float b2 = t > -64.0f ? -0.042f : -0.112f;
            const float k1 = gfk_exp((t * b2 + a2) + gfk_log(ww) * 1.2372f);
            const float k2 = (t * -0.008f + 0.0231f) + gfk_log(ww) * 0.0739f;
            ni = gfk_pow(na, k2) * k1;
        } else {
            const float k1 = gfk_exp(((gfk_log(ww) * 0.0217f + 0.0111f) * t
                                      + 1.282f) + gfk_log(ww) * 2.312f);
            const float k2 = (t * -0.0156f + -0.3949f) + gfk_log(ww) * 0.120f;
            ni = gfk_pow(na, k2) * k1;
        }
        ni = ni < na ? ni : na;
    }
    return ni;
}

// mdm_prescribed_nucleati: ice nucleated (kg-1).  WRF writes a message and
// STOPs when the number (cm-3) is above 9999 or negative; the device sets
// *stop and returns 0, the number WRF assigns before it stops, and the
// launcher raises.
__device__ float ma_nucleati(float wbar, float tair, float relhum,
                             float relhumi, float qc, float rhoair,
                             const float* naero, int* stop)
{
    const float so4_num = naero[0] * 1.0e-6f;
    const float soot_num = naero[9] * 1.0e-6f;
    const float dst_num = ((naero[2] * 1.0e-6f + naero[3] * 1.0e-6f)
                           + naero[4] * 1.0e-6f) + naero[5] * 1.0e-6f;
    float ni = 0.0f;
    const float tc = tair - 273.15f;
    if (so4_num >= 1.0e-10f && soot_num + dst_num >= 1.0e-10f) {
        const float subgrid = 1.2f;
        if (tc <= -35.0f && relhumi * subgrid >= 1.2f) {
            const float a = 12.884f - gfk_log(soot_num + dst_num) * 1.4938f;
            const float b = -(gfk_log(soot_num + dst_num) * 10.41f) - 67.69f;
            const float regm = gfk_log(wbar) * a + b;
            float n1;
            if (tc > regm) {
                if (tc < -40.0f && wbar > 1.0f) {
                    n1 = ma_hf(tc, wbar, relhum, subgrid, so4_num);
                } else {
                    n1 = ma_hetero(tc, wbar, soot_num + dst_num);
                }
            } else if (regm - 5.0f > tc) {
                n1 = ma_hf(tc, wbar, relhum, subgrid, so4_num);
            } else if (tc < -40.0f && wbar > 1.0f) {
                n1 = ma_hf(tc, wbar, relhum, subgrid, so4_num);
            } else {
                const float nihf = ma_hf(regm - 5.0f, wbar, relhum, subgrid,
                                         so4_num);
                const float niimm = ma_hetero(regm, wbar, soot_num + dst_num);
                const float nidep = 0.0f;
                if (niimm + nidep >= nihf) {
                    n1 = nihf;
                } else {
                    n1 = (niimm + nidep) * gfk_pow(FDIV(niimm + nidep, nihf),
                                                   FDIV(tc - regm, 5.0f));
                }
            }
            ni = n1;
        }
    }
    float nimey;
    if (tc < 0.0f && tc > -37.0f && qc > 1.0e-12f) {
        const float deles = relhumi > 1.5f ? 1.5f : relhumi;
        nimey = gfk_exp((deles - 1.0f) * 12.96f - 0.639f) * 1.0e-3f;
    } else {
        nimey = 0.0f;
    }
    float nuci = ni + nimey;
    if (nuci > 9999.0f || nuci < 0.0f) {
        atomicOr(stop, 1);
        nuci = 0.0f;
    }
    return FDIV(nuci * 1.0e6f, rhoair);
}
