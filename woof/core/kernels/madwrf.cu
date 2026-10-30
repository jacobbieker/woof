// MAD-WRF (WRF v4.7.1, Pedro A. Jimenez): the cloud initialization and the
// hydrometeor-tracer nudge.
//
// AUTHORITY: WRF v4.7.1 (f52c197ed)
//   phys/module_madwrf.F (745e020a...)        Init_madwrf_clouds (:71-204) with
//       Calc_cldtopz_from_brtemp (:23-69), cal_cldfra3_madwrf (:241-558),
//       find_k_from_z_agl (:560-584), find_thresh_k_downward/upward
//       (:586-626), find_cloudLayers (:633-797), Calc_tropo_height
//       (:799-868), adjust_cloudIce/H2O (:870-948), adjust_cloudFinal
//       (:955-992) and phys/module_mp_thompson.F's RSLF/RSIF;
//   dyn_em/module_big_step_utilities_em.F (bd177b6b...) Cloud_tracer_nudge
//       (:6711-6756);
// graded word for word by tools/madwrf_wrf471_oracle (gfortran -O0
// -ffp-contract=off, glibc 2.43, x86-64).
//
// Every float32 operation keeps Fortran's order and rounds once
// (FADD/FMUL/FDIV/FSUB; the unit compiles --fmad=false --ftz=false).  ``**``
// with a real exponent is glibc's powf (gfk_pow), SQRT is IEEE.  MAX/MIN of
// two reals: the ties these routines can meet are between equal finite
// values, which no operand order tells apart.
//
// Layout: one thread per column; every column array is (ncol, nk) with level
// 1 (the surface) first.  nk = kte - 1 mass levels; the w-level arrays (ph,
// phb) and CLDFRA are (ncol, nk + 1), as real.exe's kte = kde.  ``work`` is
// caller scratch of 8*(nk+1) floats per column.
//
// One departure: real.exe hands find_k_from_z_agl a dz of kte - 1 elements
// declared kts:kte, so its last iteration reads one word past DZS.  That
// iteration only matters for a height above the model top; here it reads 0,
// for which it never selects a level (the oracle's fixtures keep every
// observed height below the top).

#define MW(a, k) (a)[(k) - 1]

__device__ __forceinline__ float mw_max(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ float mw_min(float a, float b) { return a < b ? a : b; }

// module_mp_thompson.F RSLF / RSIF (the cldfra3.cu transcriptions).
__device__ float mw_rslf(float p, float t)
{
    const float c0 = .611583699E03f, c1 = .444606896E02f, c2 = .143177157E01f;
    const float c3 = .264224321E-1f, c4 = .299291081E-3f, c5 = .203154182E-5f;
    const float c6 = .702620698E-8f, c7 = .379534310E-11f, c8 = -.321582393E-13f;
    const float x = mw_max(-80.f, FSUB(t, 273.16f));
    float esl = FADD(c0, FMUL(x, FADD(c1, FMUL(x, FADD(c2, FMUL(x, FADD(c3,
        FMUL(x, FADD(c4, FMUL(x, FADD(c5, FMUL(x, FADD(c6, FMUL(x, FADD(c7,
        FMUL(x, c8))))))))))))))));
    esl = mw_min(esl, FMUL(p, 0.15f));
    return FDIV(FMUL(.622f, esl), FSUB(p, esl));
}

__device__ float mw_rsif(float p, float t)
{
    const float c0 = .609868993E03f, c1 = .499320233E02f, c2 = .184672631E01f;
    const float c3 = .402737184E-1f, c4 = .565392987E-3f, c5 = .521693933E-5f;
    const float c6 = .307839583E-7f, c7 = .105785160E-9f, c8 = .161444444E-12f;
    const float x = mw_max(-80.f, FSUB(t, 273.16f));
    float esi = FADD(c0, FMUL(x, FADD(c1, FMUL(x, FADD(c2, FMUL(x, FADD(c3,
        FMUL(x, FADD(c4, FMUL(x, FADD(c5, FMUL(x, FADD(c6, FMUL(x, FADD(c7,
        FMUL(x, c8))))))))))))))));
    esi = mw_min(esi, FMUL(p, 0.15f));
    return FDIV(FMUL(.622f, esi), mw_max(1.e-4f, FSUB(p, esi)));
}

// Calc_tropo_height.  ``theta`` scratch; returns k_tropo, *tropo_z.
__device__ int mw_tropo(const float* t, const float* p, const float* dz,
                        int kts, int kte, float* theta, float* tropo_z)
{
    const float rcp = FDIV(287.05f, 1004.f);
    const float lapse = FDIV(10.f, 1500.f);
    int k_p200 = 0, k;
    for (k = kte - 1; k >= kts; --k) {
        MW(theta, k) = FMUL(MW(t, k), gfk_pow(FDIV(100000.0f, MW(p, k)), rcp));
        if (MW(p, k) > 19999.0f && k_p200 == 0) k_p200 = k;
    }
    if ((kte - k_p200) < 3) k_p200 = kte - 3;
    for (k = k_p200 - 2; k >= kts; --k) {
        const float th1 = MW(theta, k), th2 = MW(theta, k + 2);
        const float delz = FADD(FADD(FMUL(0.5f, MW(dz, k)), MW(dz, k + 1)),
                                FMUL(0.5f, MW(dz, k + 2)));
        if (FDIV(FSUB(th2, th1), delz) < lapse || MW(p, k) > 70000.f) break;
    }
    int top = k + 2 < kte - 1 ? k + 2 : kte - 1;
    int k_tropo = kts + 2 > top ? kts + 2 : top;
    if (k_tropo > k_p200) {
        for (k = kte - 3; k >= k_p200 - 2; --k) {
            const float th1 = MW(theta, k), th2 = MW(theta, k + 2);
            const float delz = FADD(FADD(FMUL(0.5f, MW(dz, k)), MW(dz, k + 1)),
                                    FMUL(0.5f, MW(dz, k + 2)));
            if (FDIV(FSUB(th2, th1), delz) < lapse && MW(p, k) > 9500.f) break;
        }
        top = k + 2 < kte - 1 ? k + 2 : kte - 1;
        k_tropo = k_p200 - 1 > top ? k_p200 - 1 : top;
    }
    float s = 0.f;
    for (k = kts; k <= k_tropo; ++k) s = FADD(s, MW(dz, k));
    *tropo_z = s;
    return k_tropo;
}

// adjust_cloudIce (ice = 1) / adjust_cloudH2O (ice = 0).
__device__ void mw_adjust(int ice, const float* cfr, float* q, const float* qs,
                          const float* qvs, const float* t, const float* dz,
                          float entr, int k1, int k2)
{
    float tdz = 0.f;
    for (int k = k1; k <= k2; ++k) tdz = FADD(tdz, MW(dz, k));
    float maxw = fabsf(FSUB(MW(qvs, k2), MW(qvs, k1)));
    for (int k = k1; k <= k2; ++k) {
        const float have = ice ? FADD(MW(q, k), MW(qs, k)) : MW(q, k);
        maxw = mw_max(1.E-6f, FSUB(maxw, have));
    }
    maxw = mw_min(1.E-3f, maxw);
    const float tmin = ice ? 203.16f : 253.16f;
    float this_dz = 0.0f;
    for (int k = k1; k <= k2; ++k) {
        if (k == k1) this_dz = FADD(this_dz, FMUL(0.5f, MW(dz, k)));
        else this_dz = FADD(this_dz, MW(dz, k));
        const float this_w = FDIV(FMUL(maxw, this_dz), tdz);
        const float w = mw_max(1.E-6f, FMUL(this_w, FSUB(1.f, entr)));
        const float c = MW(cfr, k);
        if (c > 0.0f && c < 1.0f && MW(t, k) >= tmin)
            MW(q, k) = FADD(MW(q, k), FMUL(FMUL(c, c), w));
    }
}

// find_cloudLayers (MAD-WRF's), called without ktropo.  1 = wrf_error_fatal.
__device__ int mw_find_cloud_layers(const float* qvs, float* cfr, const float* t,
                                    const float* p, const float* dz, float entrmnt,
                                    float* qc, float* qi, const float* qs,
                                    int kts, int kte, int keep_below_lcl,
                                    float* theta, float* theta2)
{
    const float rcp = FDIV(287.05f, 1004.f);
    int k_m12C = 0, k_m40C = 0, k;
    for (k = kte - 1; k >= kts; --k) {
        MW(theta, k) = FMUL(MW(t, k), gfk_pow(FDIV(100000.0f, MW(p, k)), rcp));
        if (FSUB(MW(t, k), 273.16f) > -40.0f && MW(p, k) > 7000.0f)
            k_m40C = k_m40C > k ? k_m40C : k;
        if (FSUB(MW(t, k), 273.16f) > -12.0f && MW(p, k) > 10100.0f)
            k_m12C = k_m12C > k ? k_m12C : k;
    }
    if (k_m40C <= kts) k_m40C = kts;
    if (k_m12C <= kts) k_m12C = kts;
    if (k_m40C > kte - 2 || k_m12C > kte - 3) return 1;
    float tropo_z;
    const int k_tropo = mw_tropo(t, p, dz, kts, kte, theta2, &tropo_z);

    for (k = k_tropo + 1; k <= kte - 1; ++k) {
        const float c = MW(cfr, k);
        if (c > 0.0f && c < 1.0f) MW(cfr, k) = 0.f;
    }
    int kbot = kts + 2;
    for (k = kbot; k <= k_m12C; ++k)
        if (FSUB(MW(theta, k), MW(theta, k - 1)) > FMUL(0.010E-3f, MW(dz, k))) break;
    kbot = kts + 1 > k - 2 ? kts + 1 : k - 2;
    if (!keep_below_lcl) {
        for (k = kts; k <= kbot; ++k) {
            const float c = MW(cfr, k);
            if (c > 0.0f && c < 1.0f) MW(cfr, k) = FMUL(0.5f, c);
        }
    } else {
        kbot = kts + 1;
    }

    int k_cldb = k_tropo, k_cldt;
    k = k_tropo;
    while (k > k_m12C + 1) {
        k_cldt = 0;
        bool in_cloud = false;
        if (MW(cfr, k) >= 0.01f) { in_cloud = true; k_cldt = k_cldt > k ? k_cldt : k; }
        if (in_cloud) {
            for (int k2 = k_cldt - 1; k2 >= k_m12C; --k2)
                if (MW(cfr, k2) < 0.01f || k2 == k_m12C) { k_cldb = k2 + 1; break; }
        }
        if ((k_cldt - k_cldb + 1) >= 2) {
            mw_adjust(1, cfr, qi, qs, qvs, t, dz, entrmnt, k_cldb, k_cldt);
            k = k_cldb;
        } else if ((k_cldt - k_cldb + 1) == 1) {
            const float c = MW(cfr, k_cldb);
            if (c > 0.f && c < 1.f)
                MW(qi, k_cldb) = FMUL(FMUL(FMUL(0.05f, MW(qvs, k_cldb)), c), c);
            k = k_cldb;
        }
        k = k - 1;
    }

    k_cldb = k_m12C + 3;
    k = k_m12C + 2;
    while (k > kbot) {
        k_cldt = 0;
        bool in_cloud = false;
        if (MW(cfr, k) >= 0.01f) { in_cloud = true; k_cldt = k_cldt > k ? k_cldt : k; }
        if (in_cloud) {
            for (int k2 = k_cldt - 1; k2 >= kbot; --k2)
                if (MW(cfr, k2) < 0.01f || k2 == kbot) { k_cldb = k2 + 1; break; }
        }
        if ((k_cldt - k_cldb + 1) >= 2) {
            mw_adjust(0, cfr, qc, qs, qvs, t, dz, entrmnt, k_cldb, k_cldt);
            k = k_cldb;
        } else if ((k_cldt - k_cldb + 1) == 1) {
            const float c = MW(cfr, k_cldb);
            if (c > 0.f && c < 1.f)
                MW(qc, k_cldb) = FMUL(FMUL(FMUL(0.05f, MW(qvs, k_cldb)), c), c);
            k = k_cldb;
        }
        k = k - 1;
    }
    return 0;
}

// find_k_from_z_agl over kts..kte-1 (see the header); k_lev keeps its value
// when no layer holds z_agl, as the intent(out) integer does in WRF.
__device__ void mw_find_k(float z_agl, int* k_lev, const float* dz, int kts, int kte)
{
    float z_full = 0.0f;
    for (int k = kts; k <= kte - 1; ++k) {
        const float z_this = z_full;
        const float z_next = FADD(z_this, MW(dz, k));
        if (z_agl > 0.0f && z_agl > z_this && z_agl <= z_next) *k_lev = k;
        z_full = z_next;
    }
}

// cal_cldfra3_madwrf, modify_qvapor = .true., max_relh = 1.5, debug off,
// k_tropo absent; mode 0 none, 1 cldmask, 2 + cldtopz, 3 + cldbasez.
// Returns 1 where WRF stops.
__device__ int mw_cal_cldfra3(float* cldfra, float* qv, float* qc, float* qi,
                              float* qs, const float* dz, const float* p,
                              const float* t, float xland, float gridkm,
                              int kts, int kte, int mode, float cldmask,
                              float cldtopz, float cldbasez, float* qvs,
                              float* rh, float* rhoa, float* theta, float* theta2)
{
    const float entrmnt = 0.5f;
    const float CLDFRA_DEF = 0.5f, CLDFRA_THRESH = 0.0f, RH_NOCLOUD = 0.3f;
    const int impose_cldmask = mode == 1;
    const int impose_cldtopz = mode == 2;
    const int impose_cldbasetopz = mode == 3;
    int keep_below_lcl = 0;
    int cldtopk = -999, cldbasek = -999, cldfra_thresh_k = -999;
    int k;

    if (impose_cldmask && cldmask == 0.0f) {
        for (k = kts; k <= kte - 1; ++k) {
            MW(cldfra, k) = 0.f; MW(qc, k) = 0.f; MW(qi, k) = 0.f; MW(qs, k) = 0.f;
        }
        return 0;
    }
    if (impose_cldtopz || impose_cldbasetopz) {
        if (cldmask == 0.0f) {
            for (k = kts; k <= kte - 1; ++k) {
                MW(cldfra, k) = 0.f; MW(qc, k) = 0.f; MW(qi, k) = 0.f; MW(qs, k) = 0.f;
            }
            return 0;
        } else if (cldmask > 0.0f && cldmask <= 1.0f && cldtopz > 0.0f) {
            mw_find_k(cldtopz, &cldtopk, dz, kts, kte);
            if (cldtopk > 0 && cldtopk < kte) {
                for (k = cldtopk + 1; k <= kte - 1; ++k) {
                    MW(cldfra, k) = 0.f; MW(qc, k) = 0.f; MW(qi, k) = 0.f; MW(qs, k) = 0.f;
                }
            }
            // CLDTHICK_DEF = -999.9: the default-thickness branch is dead.
        }
    }
    if (impose_cldbasetopz) {
        if (cldmask > 0.0f && cldmask <= 1.0f && cldbasez > 0.0f) {
            mw_find_k(cldbasez, &cldbasek, dz, kts, kte);
            if (cldtopk < cldbasek) cldbasek = -999;
            if (cldbasek > 0 && cldbasek > kts) {
                for (k = kts; k <= cldbasek - 1; ++k) {
                    MW(cldfra, k) = 0.f; MW(qc, k) = 0.f; MW(qi, k) = 0.f; MW(qs, k) = 0.f;
                }
            }
        }
    }

    for (k = kts; k <= kte - 1; ++k) {
        MW(cldfra, k) = 0.0f;
        float tc = FSUB(MW(t, k), 273.15f);
        const float qvsw = mw_rslf(MW(p, k), MW(t, k));
        const float qvsi = tc < 0.0f ? mw_rsif(MW(p, k), FADD(MW(t, k), 0.025f)) : qvsw;
        float q;
        if (tc >= -12.0f) q = qvsw;
        else if (tc < -35.0f) q = qvsi;
        else q = FSUB(qvsw, FDIV(FMUL(FSUB(qvsw, qvsi), FSUB(-12.0f, tc)), 23.0f));
        if (MW(qc, k) > 1.E-8f) { MW(qv, k) = mw_max(MW(qv, k), qvsw); q = qvsw; }
        if (MW(qc, k) <= 1.E-8f && MW(qi, k) >= 1.E-9f) { MW(qv, k) = mw_max(MW(qv, k), qvsi); q = qvsi; }
        MW(qvs, k) = q;
        MW(rh, k) = mw_max(0.01f, FDIV(MW(qv, k), q));
        MW(rhoa, k) = FDIV(MW(p, k), FMUL(287.0f, MW(t, k)));
    }

    if ((impose_cldtopz || impose_cldbasetopz) && FSUB(xland, 1.5f) > 0.0f) {
        if (cldtopk > 0 && cldtopk < kte) {
            if (cldbasek < 0 || cldbasek > cldtopk) {
                cldbasek = MW(rh, cldtopk) < RH_NOCLOUD ? cldtopk + 1 : cldtopk;
            } else {
                float cldbasek_tmp = (float)(cldtopk + 1);      // a REAL in WRF
                for (k = cldtopk; k >= kts; --k) {
                    if (MW(rh, k) < RH_NOCLOUD) break;
                    cldbasek_tmp = (float)k;
                }
                if (cldbasek_tmp > (float)cldbasek) cldbasek = (int)cldbasek_tmp;
            }
        }
    }

    for (k = kts; k <= kte - 1; ++k) {
        const float delz = mw_max(100.f, MW(dz, k));
        const float g2 = FMUL(FMUL(FMUL(gridkm, gridkm), delz), 0.01f);
        const float rh_00l = FADD(0.65f, FSQRT(FDIV(1.f, FADD(25.0f, g2))));
        const float rh_00o = FADD(0.81f, FSQRT(FDIV(1.f, FADD(50.0f, g2))));
        float rhum = MW(rh, k);
        float c;
        if (MW(qc, k) > 1.E-7f || MW(qi, k) >= 1.E-7f
                || (MW(qs, k) > 1.E-6f && MW(t, k) < 273.f)) {
            c = 1.0f;
        } else {
            float rh_00 = FSUB(xland, 1.5f) > 0.f ? rh_00o : rh_00l;
            const float tc = FSUB(MW(t, k), 273.15f);
            if (tc < -12.0f) rh_00 = rh_00l;
            if (tc >= 25.0f) {
                c = 0.0f;
            } else if (tc >= -12.0f) {
                rhum = mw_min(MW(rh, k), 1.0f);
                c = mw_max(0.f, FSUB(1.0f, FSQRT(FDIV(FSUB(1.005f, rhum),
                                                      FSUB(1.005f, rh_00)))));
            } else {
                // max_relh = 1.5 > 1.12: the HRRR arm.
                rhum = mw_min(MW(rh, k), 1.45f);
                rh_00 = FADD(rh_00, FDIV(FMUL(FSUB(1.45f, rh_00), FSUB(-12.0f, tc)), 88.0f));
                c = mw_max(0.f, FSUB(1.0f, FSQRT(FDIV(FSUB(1.5f, rhum),
                                                      FSUB(1.5f, rh_00)))));
            }
            if (c > 0.f) c = mw_max(0.01f, mw_min(c, 0.9f));
        }
        MW(cldfra, k) = c;
    }

    if ((impose_cldtopz || impose_cldbasetopz) && cldtopk > 0 && cldtopk < kte) {
        for (k = cldtopk + 1; k <= kte; ++k) MW(cldfra, k) = 0.0f;
        keep_below_lcl = 1;
        for (k = cldtopk; k >= kts; --k)
            if (MW(cldfra, k) > CLDFRA_THRESH) { cldfra_thresh_k = k; break; }
        if (cldfra_thresh_k > 0) {
            if (cldfra_thresh_k < cldtopk)
                for (k = cldfra_thresh_k + 1; k <= cldtopk; ++k) MW(cldfra, k) = CLDFRA_DEF;
        }
    }
    if (impose_cldbasetopz && cldbasek > kts && cldbasek <= kte) {
        for (k = kts; k <= cldbasek - 1; ++k) MW(cldfra, k) = 0.0f;
        cldfra_thresh_k = -999;
        for (k = cldbasek; k <= kte; ++k)
            if (MW(cldfra, k) > CLDFRA_THRESH) { cldfra_thresh_k = k; break; }
        if (cldfra_thresh_k > cldbasek)
            for (k = cldbasek; k <= cldfra_thresh_k - 1; ++k)
                if (MW(rh, k) > RH_NOCLOUD) MW(cldfra, k) = CLDFRA_DEF;
    }

    if (mw_find_cloud_layers(qvs, cldfra, t, p, dz, entrmnt, qc, qi, qs, kts, kte,
                             keep_below_lcl, theta, theta2))
        return 1;

    // adjust_cloudFinal
    float lwp = 0.f, iwp = 0.f;
    for (k = kts; k <= kte - 1; ++k) {
        const float c = MW(cldfra, k);
        if (c > 0.0f && c < 1.0f) {
            lwp = FADD(lwp, FMUL(FMUL(MW(qc, k), MW(rhoa, k)), MW(dz, k)));
            iwp = FADD(iwp, FMUL(FMUL(MW(qi, k), MW(rhoa, k)), MW(dz, k)));
        }
    }
    if (lwp > 1.0f) {
        const float xfac = FDIV(1.0f, lwp);
        for (k = kts; k <= kte - 1; ++k) {
            const float c = MW(cldfra, k);
            if (c > 0.0f && c < 1.0f) MW(qc, k) = FMUL(MW(qc, k), xfac);
        }
    }
    if (iwp > 1.0f) {
        const float xfac = FDIV(1.0f, iwp);
        for (k = kts; k <= kte - 1; ++k) {
            const float c = MW(cldfra, k);
            if (c > 0.0f && c < 1.0f) MW(qi, k) = FMUL(MW(qi, k), xfac);
        }
    }
    for (k = kts; k <= kte - 1; ++k) {
        const float c = MW(cldfra, k);
        if (c > 0.20f && c < 1.0f) MW(qv, k) = mw_max(MW(qv, k), MW(qvs, k));
    }
    return 0;
}

// Calc_cldtopz_from_brtemp.
__device__ float mw_cldtopz_from_brtemp(float cldmask, const float* ts,
                                        const float* dzs, float brtemp,
                                        float tropoz, float ht, int kts, int kte)
{
    const float MISSING = -999.9f;
    float r = MISSING;
    if (brtemp == MISSING) return r;
    if (cldmask > 0.0f) {
        float h = 0.f;
        for (int k = kts; k <= kte - 1; ++k) h = FADD(h, MW(dzs, k));
        h = FADD(h, ht);
        for (int k = kte - 2; k >= kts; --k) {
            h = FSUB(h, MW(dzs, k + 1));
            if (h > tropoz) continue;
            if (MW(ts, k) > brtemp) {
                r = FADD(h, FDIV(FMUL(MW(dzs, k), FSUB(brtemp, MW(ts, k))),
                                 FSUB(MW(ts, k + 1), MW(ts, k))));
                if (r < h || r > FADD(h, MW(dzs, k))) r = tropoz;
                break;
            }
        }
    }
    return r;
}

// Init_madwrf_clouds for one column (no Skip_middle_points_t: real.exe's
// first time has hold_ups = .true., module_initialize_real.F:436).
// insert_clouds = 0..3 from the flags.  status 1 = wrf_error_fatal.
extern "C" __global__ void madwrf_cloud_init_columns(
    int ncol, int nk, float p00, float gridkm, int insert_clouds,
    int flag_cldtopz, int flag_brtemp,
    const float* __restrict__ t2_all, const float* __restrict__ p_all,
    const float* __restrict__ ph_all, const float* __restrict__ phb_all,
    float* __restrict__ qv_all, float* __restrict__ qc_all,
    float* __restrict__ qi_all, float* __restrict__ qs_all,
    const float* __restrict__ xland, const float* __restrict__ ht,
    float* __restrict__ cldmask, float* __restrict__ cldtopz,
    const float* __restrict__ cldbasez, const float* __restrict__ brtemp,
    float* __restrict__ cldfra_all, float* __restrict__ work,
    int* __restrict__ status)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const int kts = 1, kte = nk + 1;
    const size_t m = (size_t)col * nk, w = (size_t)col * (nk + 1);
    const float* t2 = t2_all + m;
    const float* p = p_all + m;
    const float* ph = ph_all + w;
    const float* phb = phb_all + w;
    float* qv = qv_all + m;
    float* qc = qc_all + m;
    float* qi = qi_all + m;
    float* qs = qs_all + m;
    float* cldfra = cldfra_all + w;
    float* ts = work + (size_t)col * 8 * (nk + 1);
    float* dzs = ts + (nk + 1);
    float* qvs = dzs + (nk + 1);
    float* rh = qvs + (nk + 1);
    float* rhoa = rh + (nk + 1);
    float* theta = rhoa + (nk + 1);
    float* theta2 = theta + (nk + 1);
    float* theta3 = theta2 + (nk + 1);
    const float mw_t0 = 300.f, mw_g = 9.81f;
    const float mw_rcp = FDIV(287.f, FDIV(FMUL(7.f, 287.f), 2.f));
    const float MISSING_CLDTOPZ = -999.9f, MISSING_CLDMASK = -999.9f;

    for (int k = kts; k <= kte - 1; ++k) {
        MW(ts, k) = FDIV(FADD(MW(t2, k), mw_t0), gfk_pow(FDIV(p00, MW(p, k)), mw_rcp));
        MW(dzs, k) = FDIV(FSUB(FADD(MW(ph, k + 1), MW(phb, k + 1)),
                               FADD(MW(ph, k), MW(phb, k))), mw_g);
    }
    MW(ts, kte) = 0.f;
    MW(dzs, kte) = 0.f;
    float tropoz;
    mw_tropo(ts, p, dzs, kts, kte, theta3, &tropoz);
    tropoz = FADD(tropoz, ht[col]);

    float cldtopz_agl = MISSING_CLDTOPZ;
    if (insert_clouds > 0) {
        if (flag_cldtopz == 1) {
            const float z = cldtopz[col];
            if (z > 0.0f) cldmask[col] = 1.0f;
            else if (z == 0.0f) cldmask[col] = 0.0f;
            else cldmask[col] = MISSING_CLDMASK;
            if (z > 0.0f) cldtopz_agl = mw_max(0.0f, FSUB(z, ht[col]));
        } else if (flag_brtemp == 1) {
            cldtopz[col] = mw_cldtopz_from_brtemp(cldmask[col], ts, dzs, brtemp[col],
                                                  tropoz, ht[col], kts, kte);
            if (cldtopz[col] > 0.0f) cldtopz_agl = mw_max(0.0f, FSUB(cldtopz[col], ht[col]));
        }
    }
    status[col] = mw_cal_cldfra3(cldfra, qv, qc, qi, qs, dzs, p, ts, xland[col],
                                 gridkm, kts, kte, insert_clouds, cldmask[col],
                                 cldtopz_agl, cldbasez[col], qvs, rh, rhoa,
                                 theta, theta2);
}

// Cloud_tracer_nudge (module_big_step_utilities_em.F:6711-6756) over the
// tile [i0, i1) x [j0, j1) of every level; arrays (nk, ny, nx).  The caller
// tests xtime < dt_nudge.
extern "C" __global__ void madwrf_nudge(
    int nk, int ny, int nx, int i0, int i1, int j0, int j1, float dt,
    float dt_relax, float* __restrict__ qc, float* __restrict__ qi,
    float* __restrict__ qs, const float* __restrict__ trqc,
    const float* __restrict__ trqi, const float* __restrict__ trqs)
{
    const int ni = i1 - i0, nj = j1 - j0;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni <= 0 || nj <= 0 || t >= ni * nj * nk) return;
    const int i = i0 + t % ni, j = j0 + (t / ni) % nj, k = t / (ni * nj);
    const size_t x = ((size_t)k * ny + j) * nx + i;
    qc[x] = FADD(qc[x], FDIV(FMUL(FSUB(trqc[x], qc[x]), dt), dt_relax));
    qi[x] = FADD(qi[x], FDIV(FMUL(FSUB(trqi[x], qi[x]), dt), dt_relax));
    qs[x] = FADD(qs[x], FDIV(FMUL(FSUB(trqs[x], qs[x]), dt), dt_relax));
}
