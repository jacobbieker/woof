// WRF-Solar cloud fraction, icloud = 3 (G. Thompson, 2014Oct31).
//
// Transcribes WRF v4.7.1 phys/module_radiation_driver.F (f52c197ed)
//   cal_cldfra3        :4003-4150
//   find_cloudLayers   :4157-4341
//   adjust_cloudIce    :4345-4383
//   adjust_cloudH2O    :4387-4425
//   adjust_cloudFinal  :4432-4469
// and phys/module_mp_thompson.F RSLF/RSIF (:5383-5444), as the radiation
// driver calls them (:1462-1510): one column at a time, modify_qvapor =
// .false., max_relh = 1.5, debug_flag = .false.  The routine rewrites the
// radiation's copies of qc and qi (the driver restores the prognostic ones
// afterwards, :3137-3155) and the driver's CLDFRA; qs and qv are read only.
//
// Every float32 operation rounds once, in Fortran's left-to-right order
// (FADD/FMUL/... from glibc_flt32.cuh; the unit also compiles with
// --fmad=false --ftz=false).  LOG10 and ** are glibc 2.43's log10f (CORE-MATH,
// correctly rounded: cm_log10f, corem_log10f.cuh) and powf (Arm, unchanged
// since glibc 2.28: gfk_pow), the libm of the repository's WRF v4.7.1 column
// oracles; CUDA's own log10f/powf round differently, and so does glibc
// 2.39's FDLIBM log10f (gfk_log10).  The reference is gfortran -O0
// -ffp-contract=off (tools/cldfra3_wrf471_oracle).
//
// Layout: every profile is (ncol, nz), level k = 1 (surface) .. nz
// contiguous within a column.  One thread per column.  ``work`` is caller
// scratch of 4 * ncol * nz floats (qvs, rh, rhoa, theta); no local arrays.
// ``status`` receives 1 for a column where WRF stops in wrf_error_fatal
// ('FATAL ERROR, problem in temperature profile.': the -12 C level sits
// within three levels of the model top); that column's outputs are left as
// cal_cldfra3 had them at the stop.

__device__ __forceinline__ float c3_max(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ float c3_min(float a, float b) { return a < b ? a : b; }

// module_mp_thompson.F RSLF: liquid saturation mixing ratio (Flatau 1992).
__device__ float c3_rslf(float p, float t)
{
    const float c0 = .611583699E03f, c1 = .444606896E02f, c2 = .143177157E01f;
    const float c3 = .264224321E-1f, c4 = .299291081E-3f, c5 = .203154182E-5f;
    const float c6 = .702620698E-8f, c7 = .379534310E-11f, c8 = -.321582393E-13f;
    const float x = c3_max(-80.f, FSUB(t, 273.16f));
    float esl = FADD(c0, FMUL(x, FADD(c1, FMUL(x, FADD(c2, FMUL(x, FADD(c3,
        FMUL(x, FADD(c4, FMUL(x, FADD(c5, FMUL(x, FADD(c6, FMUL(x, FADD(c7,
        FMUL(x, c8))))))))))))))));
    esl = c3_min(esl, FMUL(p, 0.15f));
    return FDIV(FMUL(.622f, esl), FSUB(p, esl));
}

// module_mp_thompson.F RSIF: ice saturation mixing ratio.
__device__ float c3_rsif(float p, float t)
{
    const float c0 = .609868993E03f, c1 = .499320233E02f, c2 = .184672631E01f;
    const float c3 = .402737184E-1f, c4 = .565392987E-3f, c5 = .521693933E-5f;
    const float c6 = .307839583E-7f, c7 = .105785160E-9f, c8 = .161444444E-12f;
    const float x = c3_max(-80.f, FSUB(t, 273.16f));
    float esi = FADD(c0, FMUL(x, FADD(c1, FMUL(x, FADD(c2, FMUL(x, FADD(c3,
        FMUL(x, FADD(c4, FMUL(x, FADD(c5, FMUL(x, FADD(c6, FMUL(x, FADD(c7,
        FMUL(x, c8))))))))))))))));
    esi = c3_min(esi, FMUL(p, 0.15f));
    return FDIV(FMUL(.622f, esi), c3_max(1.e-4f, FSUB(p, esi)));
}

// 1-based Fortran indexing into one column.
#define C3(a, k) (a)[(k) - 1]

// adjust_cloudIce (is_ice = 1, threshold 203.16 K, qi and qs) and
// adjust_cloudH2O (is_ice = 0, threshold 258.16 K, qc), which are the same
// statements on different arrays.
__device__ void c3_adjust(int is_ice, const float* cfr, float* q,
                          const float* qs, const float* qvs, const float* t,
                          const float* dz, float entr, int k1, int k2)
{
    float tdz = 0.f;
    for (int k = k1; k <= k2; ++k) tdz = FADD(tdz, C3(dz, k));
    float maxw = c3_max(0.0f, FSUB(C3(qvs, k1), C3(qvs, k2)));
    for (int k = k1; k <= k2; ++k) {
        const float have = is_ice ? FADD(C3(q, k), C3(qs, k)) : C3(q, k);
        maxw = c3_max(1.E-6f, FSUB(maxw, have));
    }
    maxw = c3_min(1.E-4f, maxw);
    const float tmin = is_ice ? 203.16f : 258.16f;
    float this_dz = 0.0f;
    for (int k = k1; k <= k2; ++k) {
        if (k == k1) this_dz = FADD(this_dz, FMUL(0.5f, C3(dz, k)));
        else this_dz = FADD(this_dz, C3(dz, k));
        const float this_w = FDIV(FMUL(maxw, this_dz), tdz);
        const float w = c3_max(1.E-6f, FMUL(this_w, FSUB(1.f, entr)));
        const float c = C3(cfr, k);
        if (c > 0.0f && c < 1.0f && C3(t, k) >= tmin)
            C3(q, k) = FADD(C3(q, k), FMUL(FMUL(c, c), w));
    }
}

// find_cloudLayers.  Returns 1 where WRF calls wrf_error_fatal.
__device__ int c3_find_cloud_layers(const float* qvs, float* cfr,
                                    const float* t, const float* p,
                                    const float* dz, float entrmnt,
                                    float* qc, float* qi, const float* qs,
                                    float* theta, int kts, int kte)
{
    // 287.05/1004. and 10./1500. are default-REAL constant expressions,
    // folded by gfortran to the correctly rounded float32 quotient.
    const float rcp = FDIV(287.05f, 1004.f);
    const float lapse = FDIV(10.f, 1500.f);
    int k_m12C = 0, k_p150 = 0, k;
    for (k = kte; k >= kts; --k) {
        C3(theta, k) = FMUL(C3(t, k), gfk_pow(FDIV(100000.0f, C3(p, k)), rcp));
        if (FSUB(C3(t, k), 273.16f) > -12.0f && C3(p, k) > 10100.0f)
            k_m12C = k_m12C > k ? k_m12C : k;
        if (C3(p, k) > 14999.0f && k_p150 == 0) k_p150 = k;
    }
    if (k_m12C <= kts) k_m12C = kts;
    if (k_m12C > kte - 3) return 1;

    if ((kte - k_p150) < 3) k_p150 = kte - 3;
    for (k = k_p150 - 2; k >= kts; --k) {
        const float theta1 = C3(theta, k), theta2 = C3(theta, k + 2);
        const float delz = FADD(FADD(FMUL(0.5f, C3(dz, k)), C3(dz, k + 1)),
                                FMUL(0.5f, C3(dz, k + 2)));
        if (FDIV(FSUB(theta2, theta1), delz) < lapse || C3(p, k) > 70000.f) break;
    }
    int k_tropo = kts + 2 > (k + 2 < kte - 1 ? k + 2 : kte - 1)
                ? kts + 2 : (k + 2 < kte - 1 ? k + 2 : kte - 1);
    if (k_tropo > k_p150) {
        for (k = kte - 3; k >= k_p150 - 2; --k) {
            const float theta1 = C3(theta, k), theta2 = C3(theta, k + 2);
            const float delz = FADD(FADD(FMUL(0.5f, C3(dz, k)), C3(dz, k + 1)),
                                    FMUL(0.5f, C3(dz, k + 2)));
            if (FDIV(FSUB(theta2, theta1), delz) < lapse && C3(p, k) > 9500.f) break;
        }
        const int top = k + 2 < kte - 1 ? k + 2 : kte - 1;
        k_tropo = k_p150 - 1 > top ? k_p150 - 1 : top;
    }

    // Eliminate fractional clouds above the tropopause.
    for (k = k_tropo + 1; k <= kte; ++k) {
        const float c = C3(cfr, k);
        if (c > 0.0f && c < 1.0f) C3(cfr, k) = 0.f;
    }

    // Well-mixed convective boundary layer below the LCL.
    int kbot = kts + 1;
    for (k = kbot; k <= k_m12C; ++k)
        if (FSUB(C3(theta, k), C3(theta, k - 1)) > FMUL(0.010E-3f, C3(dz, k))) break;
    kbot = kts + 1 > k - 2 ? kts + 1 : k - 2;
    for (k = kts; k <= kbot; ++k) {
        const float c = C3(cfr, k);
        if (c > 0.0f && c < 1.0f) C3(cfr, k) = c3_max(0.01f, FMUL(0.5f, c));
    }
    for (k = kts; k <= k_tropo; ++k)
        if (C3(cfr, k) > 0.0f) kbot = k < kbot ? k : kbot;

    // Ice cloud layers between the tropopause and the -12 C level.
    int k_cldb = k_tropo, k_cldt;
    k = k_tropo;
    while (k > k_m12C + 1) {
        k_cldt = 0;
        bool in_cloud = false;
        if (C3(cfr, k) >= 0.01f) { in_cloud = true; k_cldt = k_cldt > k ? k_cldt : k; }
        if (in_cloud) {
            for (int k2 = k_cldt - 1; k2 >= k_m12C; --k2) {
                if (C3(cfr, k2) < 0.01f || k2 == k_m12C) { k_cldb = k2 + 1; break; }
            }
        }
        if ((k_cldt - k_cldb + 1) >= 2) {
            c3_adjust(1, cfr, qi, qs, qvs, t, dz, entrmnt, k_cldb, k_cldt);
            k = k_cldb;
        } else if ((k_cldt - k_cldb + 1) == 1) {
            const float c = C3(cfr, k_cldb);
            if (c > 0.0f && c < 1.0f)
                C3(qi, k_cldb) = FADD(C3(qi, k_cldb), FMUL(FMUL(0.05f, C3(qvs, k_cldb)), c));
            k = k_cldb;
        }
        k = k - 1;
    }

    // Water cloud layers below.
    k_cldb = k_m12C + 3;
    k = k_m12C + 2;
    while (k > kbot) {
        k_cldt = 0;
        bool in_cloud = false;
        if (C3(cfr, k) >= 0.01f) { in_cloud = true; k_cldt = k_cldt > k ? k_cldt : k; }
        if (in_cloud) {
            for (int k2 = k_cldt - 1; k2 >= kbot; --k2) {
                if (C3(cfr, k2) < 0.01f || k2 == kbot) { k_cldb = k2 + 1; break; }
            }
        }
        if ((k_cldt - k_cldb + 1) >= 2) {
            c3_adjust(0, cfr, qc, qs, qvs, t, dz, entrmnt, k_cldb, k_cldt);
            k = k_cldb;
        } else if ((k_cldt - k_cldb + 1) == 1) {
            const float c = C3(cfr, k_cldb);
            if (c > 0.0f && c < 1.0f)
                C3(qc, k_cldb) = FADD(C3(qc, k_cldb), FMUL(FMUL(0.05f, C3(qvs, k_cldb)), c));
            k = k_cldb;
        }
        k = k - 1;
    }
    return 0;
}

// adjust_cloudFinal: rescale the made-up condensate of fractional layers
// when its column path exceeds 1 kg m-2.
__device__ void c3_adjust_final(const float* cfr, float* qc, float* qi,
                                const float* rho, const float* dz,
                                int kts, int kte)
{
    float lwp = 0.f, iwp = 0.f;
    for (int k = kts; k <= kte; ++k) {
        const float c = C3(cfr, k);
        if (c > 0.0f && c < 1.0f) {
            lwp = FADD(lwp, FMUL(FMUL(C3(qc, k), C3(rho, k)), C3(dz, k)));
            iwp = FADD(iwp, FMUL(FMUL(C3(qi, k), C3(rho, k)), C3(dz, k)));
        }
    }
    if (lwp > 1.0f) {
        const float xfac = FDIV(1.0f, lwp);
        for (int k = kts; k <= kte; ++k) {
            const float c = C3(cfr, k);
            if (c > 0.0f && c < 1.0f) C3(qc, k) = FMUL(C3(qc, k), xfac);
        }
    }
    if (iwp > 1.0f) {
        const float xfac = FDIV(1.0f, iwp);
        for (int k = kts; k <= kte; ++k) {
            const float c = C3(cfr, k);
            if (c > 0.0f && c < 1.0f) C3(qi, k) = FMUL(C3(qi, k), xfac);
        }
    }
}

extern "C" __global__ void cldfra3_columns(
    int ncol, int nz,
    const float* __restrict__ qv_all, float* __restrict__ qc_all,
    float* __restrict__ qi_all, const float* __restrict__ qs_all,
    const float* __restrict__ dz_all, const float* __restrict__ p_all,
    const float* __restrict__ t_all, const float* __restrict__ xland,
    float gridkm, float* __restrict__ cldfra_all,
    float* __restrict__ work, int* __restrict__ status)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const size_t off = (size_t)col * (size_t)nz;
    const float* qv = qv_all + off;
    float* qc = qc_all + off;
    float* qi = qi_all + off;
    const float* qs = qs_all + off;
    const float* dz = dz_all + off;
    const float* p = p_all + off;
    const float* t = t_all + off;
    float* cf = cldfra_all + off;
    float* qvs = work + 4 * off;
    float* rh = qvs + nz;
    float* rhoa = rh + nz;
    float* theta = rhoa + nz;
    const int kts = 1, kte = nz;
    const float entrmnt = 0.5f;
    const float land = xland[col];

    for (int k = kts; k <= kte; ++k) {
        C3(cf, k) = 0.0f;
        const float qvsw = c3_rslf(C3(p, k), C3(t, k));
        const float qvsi = c3_rsif(C3(p, k), C3(t, k));
        const float tc = FSUB(C3(t, k), 273.15f);
        float q;
        if (tc >= -12.0f) q = qvsw;
        else if (tc < -35.0f) q = qvsi;
        else q = FSUB(qvsw, FDIV(FMUL(FSUB(qvsw, qvsi), FSUB(-12.0f, tc)), 23.0f));
        C3(qvs, k) = q;
        C3(rh, k) = c3_max(0.01f, FDIV(C3(qv, k), q));
        C3(rhoa, k) = FDIV(C3(p, k), FMUL(287.0f, C3(t, k)));
    }

    for (int k = kts; k <= kte; ++k) {
        const float delz = c3_max(100.f, C3(dz, k));
        const float scale = FSQRT(FDIV(1.f, FADD(50.0f,
            FMUL(FMUL(FMUL(gridkm, gridkm), delz), 0.01f))));
        const float rh_00l = FADD(0.77f, c3_min(0.22f, scale));
        const float rh_00o = FADD(0.85f, c3_min(0.14f, scale));
        float rhum = C3(rh, k);
        const float qck = C3(qc, k), qik = C3(qi, k);
        float c;
        if (qck > 1.E-6f || qik >= 1.E-7f
                || (C3(qs, k) > 1.E-6f && C3(t, k) < 273.f)) {
            c = 1.0f;
        } else if (FADD(qck, qik) > 1.E-10f && FADD(qck, qik) < 1.E-6f) {
            c = c3_min(0.99f, FMUL(0.1f, FADD(11.0f, cm_log10f(FADD(qck, qik)))));
        } else {
            float rh_00 = FSUB(land, 1.5f) > 0.f ? rh_00o : rh_00l;
            const float tc = c3_max(-80.0f, FSUB(C3(t, k), 273.15f));
            if (tc < -12.0f) rh_00 = rh_00l;
            if (tc >= 25.0f) {
                c = 0.0f;
            } else if (tc >= -12.0f) {
                rhum = c3_min(C3(rh, k), 1.0f);
                c = c3_max(0.f, FSUB(1.0f, FSQRT(FDIV(FSUB(1.001f, rhum),
                                                      FSUB(1.001f, rh_00)))));
            } else {
                // max_relh = 1.5 > 1.12: the HRRR arm.
                rhum = c3_min(C3(rh, k), 1.45f);
                rh_00 = FADD(rh_00, FDIV(FMUL(FSUB(1.45f, rh_00), FSUB(-12.0f, tc)), 73.0f));
                c = c3_max(0.f, FSUB(1.0f, FSQRT(FDIV(FSUB(1.46f, rhum),
                                                      FSUB(1.46f, rh_00)))));
            }
            if (c > 0.f) c = c3_max(0.01f, c3_min(c, 0.99f));
        }
        if (c > 0.0f && C3(p, k) < 7000.0f) c = 0.0f;
        C3(cf, k) = c;
    }

    if (c3_find_cloud_layers(qvs, cf, t, p, dz, entrmnt, qc, qi, qs, theta,
                             kts, kte)) {
        status[col] = 1;
        return;
    }
    c3_adjust_final(cf, qc, qi, rhoa, dz, kts, kte);
    status[col] = 0;
}

// cm_log10f under this unit's own compile route, for the Metal/CUDA sweep in
// tests/test_cldfra3_wrf471_oracle.py (tools/corem_log10f_proof holds the
// exhaustive proofs).
extern "C" __global__ void cldfra3_log10_probe(int n, const float* __restrict__ x,
                                               float* __restrict__ y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = cm_log10f(x[i]);
}
