// WRF-Solar diagnostics (solar_diagnostics = 1): WRF v4.7.1
// phys/module_diag_solar.F (f52c197ed), SUBROUTINE solar_diag and its
// Integrate_1var/Integrate_2var, as module_diagnostics_driver.F:1181-1213
// calls it every time step (kts = 1, kte = the mass-level count, so every
// path integral and the 2-D cloud fraction stop one level below the top).
//
// Every float32 operation rounds once in Fortran's left-to-right order
// (FADD/FMUL/... from glibc_flt32.cuh; the unit compiles --fmad=false
// --ftz=false).  ACOS is glibc 2.43's acosf (CORE-MATH, correctly rounded:
// glibc_acosf, glibc_trig_flt32.cuh), the libm of the repository's WRF
// v4.7.1 column oracles.  PI = 4*ATAN(1.0) and DEGRAD = 180/PI are default-
// REAL constant expressions gfortran folds correctly rounded.  The reference
// is gfortran -O0 -ffp-contract=off (tools/solar_diag_wrf471_oracle).
//
// Layout: a mass field is (nz, ncol) with column fastest -- woof's (nz, ny,
// nx) state -- and ph/phb are (nz+1, ncol).  One thread per column.  Where
// WRF's INTENT(OUT) argument is left unassigned (the radii and optical
// depths of a scheme without that radius, has_req* = 0; WVP without qv) the
// kernel leaves the output word as it was, which is what WRF's grid array
// does.  ``flags`` bit 0..3: F_QV, F_QC, F_QI, F_QS present (WRF's
// f_qx .and. p_qx > param_first_scalar); bit 4..6: has_reqc/i/s.

#define SD_MISSING (-999.0f)

__device__ __forceinline__ float sd_max(float a, float b) { return a > b ? a : b; }

// Integrate_1var / Integrate_2var: k = kts .. kte-1, sequential.
__device__ float sd_int1(const float* rhodz, const float* v, size_t ncol,
                         int nz)
{
    float s = 0.0f;
    for (int k = 0; k < nz - 1; ++k)
        s = FADD(s, FMUL(v[(size_t)k * ncol], rhodz[(size_t)k * ncol]));
    return s;
}
__device__ float sd_int2(const float* rhodz, const float* v1, const float* v2,
                         size_t ncol, int nz)
{
    float s = 0.0f;
    for (int k = 0; k < nz - 1; ++k) {
        const size_t o = (size_t)k * ncol;
        s = FADD(s, FMUL(FMUL(v1[o], v2[o]), rhodz[o]));
    }
    return s;
}

// The water-path / radius / optical-depth group of one species.
// kind 0 = liquid (tau 3/2*LWP/re/1000), 1 = ice or snow (Matrosov 2002).
__device__ void sd_species(int kind, int has_re, const float* rhodz,
                           const float* q, const float* re, size_t ncol,
                           int nz, size_t col, float* path_out,
                           float* re_out, float* tau_out)
{
    const float q_aux = sd_int1(rhodz, q, ncol, nz);
    const float path = q_aux;
    path_out[col] = fabsf(sd_max(path, 0.0f));          // SIGN(MAX(p,0),1)
    if (!has_re) return;
    float r;
    if (q_aux > 0.0f) r = FDIV(sd_int2(rhodz, q, re, ncol, nz), q_aux);
    else r = 0.0f;
    re_out[col] = r;
    if (r > 0.0f) {
        if (kind == 0)
            tau_out[col] = FDIV(FDIV(FMUL(1.5f, path), r), 1000.0f);
        else
            tau_out[col] = FMUL(FMUL(path, 1000.0f),
                                FADD(0.02f, FDIV(4.2f, FMUL(FMUL(2.0f, r), 1.0e+6f))));
    } else {
        tau_out[col] = 0.0f;
    }
}

// Condensate of one level k (1-based) as WRF sums it: wc = 0.0, then
// + qc, + qi, + qs for the species the run carries.
__device__ float sd_wc(int fqc, int fqi, int fqs, const float* qc,
                       const float* qi, const float* qs, size_t ncol, int k)
{
    const size_t o = (size_t)(k - 1) * ncol;
    float w = 0.0f;
    if (fqc) w = FADD(w, qc[o]);
    if (fqi) w = FADD(w, qi[o]);
    if (fqs) w = FADD(w, qs[o]);
    return w;
}

// ((ph(k)+phb(k)) + (ph(k+1)+phb(k+1))) / (2.0*G), G = 9.81.
__device__ float sd_height(const float* ph, const float* phb, size_t ncol, int k)
{
    const size_t lo = (size_t)(k - 1) * ncol, hi = (size_t)k * ncol;
    return FDIV(FADD(FADD(ph[lo], phb[lo]), FADD(ph[hi], phb[hi])),
                FMUL(2.0f, 9.81f));
}

// Cloud base / top heights from a condensate column (qc, qi, qs present per
// flags), module_diag_solar.F:282-327.  WRF's own quirks are kept: a base
// found only at the top level, or a top found only at the lowest level, is
// MISSING.
__device__ void sd_base_top(int fqc, int fqi, int fqs, const float* qc,
                            const float* qi, const float* qs, const float* ph,
                            const float* phb, size_t ncol, int nz, size_t col,
                            float* base, float* top)
{
    const int kts = 1, kte = nz;
    int k = kts;
    float wc = sd_wc(fqc, fqi, fqs, qc, qi, qs, ncol, k);
    while (wc < 1E-5f && k < kte) {
        k = k + 1;
        wc = sd_wc(fqc, fqi, fqs, qc, qi, qs, ncol, k);
    }
    base[col] = (k == kte) ? SD_MISSING : sd_height(ph, phb, ncol, k);
    k = kte;
    wc = sd_wc(fqc, fqi, fqs, qc, qi, qs, ncol, k);
    while (wc < 1E-5f && k > kts) {
        k = k - 1;
        wc = sd_wc(fqc, fqi, fqs, qc, qi, qs, ncol, k);
    }
    top[col] = (k == kts) ? SD_MISSING : sd_height(ph, phb, ncol, k);
}

// Output plane order (each ncol floats): see SOLAR_OUTPUTS in
// woof/core/solar_diagnostics.py.
enum {
    SD_CLRNIDX, SD_SZA, SD_GHI_ACCUM, SD_CLDFRAC2D, SD_WVP, SD_LWP, SD_IWP,
    SD_SWP, SD_WP_SUM, SD_LWP_TOT, SD_IWP_TOT, SD_WP_TOT_SUM, SD_RE_QC,
    SD_RE_QI, SD_RE_QS, SD_RE_QC_TOT, SD_RE_QI_TOT, SD_TAU_QC, SD_TAU_QI,
    SD_TAU_QS, SD_TAU_QC_TOT, SD_TAU_QI_TOT, SD_CBASEHT, SD_CTOPHT,
    SD_CBASEHT_TOT, SD_CTOPHT_TOT, SD_NOUT
};

extern "C" __global__ void solar_diag_columns(
    int ncol_i, int nz, float dt, int flags,
    const float* __restrict__ rho, const float* __restrict__ dz8w,
    const float* __restrict__ ph, const float* __restrict__ phb,
    const float* __restrict__ cldfra, const float* __restrict__ coszen,
    const float* __restrict__ swdnb, const float* __restrict__ swdnt,
    const float* __restrict__ qv, const float* __restrict__ qc,
    const float* __restrict__ qi, const float* __restrict__ qs,
    const float* __restrict__ qc_tot, const float* __restrict__ qi_tot,
    const float* __restrict__ re_cloud, const float* __restrict__ re_ice,
    const float* __restrict__ re_snow, float* __restrict__ rhodz_work,
    float* __restrict__ out)
{
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t ncol = (size_t)ncol_i;
    if (col >= ncol) return;
    const int fqv = flags & 1, fqc = (flags >> 1) & 1, fqi = (flags >> 2) & 1,
              fqs = (flags >> 3) & 1;
    const int hrc = (flags >> 4) & 1, hri = (flags >> 5) & 1, hrs = (flags >> 6) & 1;
#define O(n) (out + (size_t)(n) * ncol)

    // SOLAR ZENITH ANGLE; PI = 4*ATAN(1.0), DEGRAD = 180/PI (folded).
    const float pi = FMUL(4.0f, __uint_as_float(0x3f490fdbu));   // atanf(1.0)
    const float degrad = FDIV(180.0f, pi);
    const float sza = FMUL(degrad, sd_max(glibc_acosf(coszen[col]), 0.0f));
    O(SD_SZA)[col] = sza;
    // CLEARNESS INDEX
    O(SD_CLRNIDX)[col] = (sza < 90.0f && swdnt[col] > 0.0f)
                         ? FDIV(swdnb[col], swdnt[col]) : SD_MISSING;
    // ACCUMULATED GHI
    O(SD_GHI_ACCUM)[col] = FADD(O(SD_GHI_ACCUM)[col], FMUL(swdnb[col], dt));
    // 2-D CLOUD FRACTION: MAXVAL(cldfra(kts:kte-1))
    float cmax = cldfra[col];
    for (int k = 1; k < nz - 1; ++k) {
        const float c = cldfra[(size_t)k * ncol + col];
        if (c > cmax) cmax = c;
    }
    O(SD_CLDFRAC2D)[col] = cmax;

    // rhodz = rho*dz8w/(1.+qv), the levels the integrals read
    float* rhodz = rhodz_work + col;
    for (int k = 0; k < nz - 1; ++k) {
        const size_t i = (size_t)k * ncol + col;
        rhodz[(size_t)k * ncol] = FDIV(FMUL(rho[i], dz8w[i]), FADD(1.0f, qv[i]));
    }
    const float* QV = qv + col;
    if (fqv) O(SD_WVP)[col] = sd_int1(rhodz, QV, ncol, nz);

    if (fqc) {
        sd_species(0, hrc, rhodz, qc + col, re_cloud + col, ncol, nz, col,
                   O(SD_LWP), O(SD_RE_QC), O(SD_TAU_QC));
        sd_species(0, hrc, rhodz, qc_tot + col, re_cloud + col, ncol, nz, col,
                   O(SD_LWP_TOT), O(SD_RE_QC_TOT), O(SD_TAU_QC_TOT));
    } else {
        O(SD_LWP)[col] = O(SD_RE_QC)[col] = O(SD_TAU_QC)[col] = SD_MISSING;
        O(SD_LWP_TOT)[col] = O(SD_RE_QC_TOT)[col] = O(SD_TAU_QC_TOT)[col] = SD_MISSING;
    }
    if (fqi) {
        sd_species(1, hri, rhodz, qi + col, re_ice + col, ncol, nz, col,
                   O(SD_IWP), O(SD_RE_QI), O(SD_TAU_QI));
        sd_species(1, hri, rhodz, qi_tot + col, re_ice + col, ncol, nz, col,
                   O(SD_IWP_TOT), O(SD_RE_QI_TOT), O(SD_TAU_QI_TOT));
    } else {
        O(SD_IWP)[col] = O(SD_RE_QI)[col] = O(SD_TAU_QI)[col] = SD_MISSING;
        O(SD_IWP_TOT)[col] = O(SD_RE_QI_TOT)[col] = O(SD_TAU_QI_TOT)[col] = SD_MISSING;
    }
    if (fqs) {
        sd_species(1, hrs, rhodz, qs + col, re_snow + col, ncol, nz, col,
                   O(SD_SWP), O(SD_RE_QS), O(SD_TAU_QS));
    } else {
        O(SD_SWP)[col] = O(SD_RE_QS)[col] = O(SD_TAU_QS)[col] = SD_MISSING;
    }

    if (fqc || fqi || fqs) {
        const float wsum = FADD(FADD(sd_max(O(SD_LWP)[col], 0.0f),
                                     sd_max(O(SD_IWP)[col], 0.0f)),
                                sd_max(O(SD_SWP)[col], 0.0f));
        O(SD_WP_SUM)[col] = wsum;
        if (wsum > 0.0f) {
            sd_base_top(fqc, fqi, fqs, qc + col, qi + col, qs + col, ph + col,
                        phb + col, ncol, nz, col, O(SD_CBASEHT), O(SD_CTOPHT));
        } else {
            O(SD_CBASEHT)[col] = O(SD_CTOPHT)[col] = SD_MISSING;
        }
        // snow from the resolved field only
        const float wtot = FADD(FADD(sd_max(O(SD_LWP_TOT)[col], 0.0f),
                                     sd_max(O(SD_IWP_TOT)[col], 0.0f)),
                                sd_max(O(SD_SWP)[col], 0.0f));
        O(SD_WP_TOT_SUM)[col] = wtot;
        if (wtot > 0.0f) {
            sd_base_top(fqc, fqi, fqs, qc_tot + col, qi_tot + col, qs + col,
                        ph + col, phb + col, ncol, nz, col, O(SD_CBASEHT_TOT),
                        O(SD_CTOPHT_TOT));
        } else {
            O(SD_CBASEHT_TOT)[col] = O(SD_CTOPHT_TOT)[col] = SD_MISSING;
        }
    } else {
        O(SD_WP_SUM)[col] = O(SD_CBASEHT)[col] = O(SD_CTOPHT)[col] = SD_MISSING;
        O(SD_WP_TOT_SUM)[col] = O(SD_CBASEHT_TOT)[col] = O(SD_CTOPHT_TOT)[col] = SD_MISSING;
    }
}
#undef O
