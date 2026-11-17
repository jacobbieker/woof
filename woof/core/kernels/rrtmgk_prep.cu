// RRTMG-K wrapper preparation (WRF 4.7.1 ra_lw_physics = ra_sw_physics = 14):
// rad_rrtmg_driver's per-column glue (phys/module_ra_rrtmg_swk.F:10174-10857),
// effectRad (phys/module_ra_effective_radius.F:66-246), the wrapper ozone
// (inirad/o3data, phys/module_ra_rrtmg_lwk.F:12010-12110) and the reduced
// McICA generator (mcica_subcol / generate_stochastic_redu,
// phys/module_ra_rrtmg_lwk.F:2192-2553).
// Third-party algorithm: Copyright (c) 2020 Atmospheric and Environmental
// Research. BSD-3-Clause; see licenses/LICENSE-AER-RRTMG-BSD-3-Clause.txt.
//
// woof/core/rrtmgk.py compiles this file after kernels/glibc_flt32.cuh
// (gfk_pow, gfk_exp) and kernels/rrtmg_mcica_wrf.cu (the kissvec seed and
// draw helpers rmcw_seed_from_play / rmcw_kiss_advance / rmcw_kiss_draw),
// through compile_using_nvrtc with --ftz=false.  Every FP32 operation
// rounds once with no contraction (inline PTX .rn, no .ftz): cloud paths
// and mixing ratios reach the subnormal range.
//
// MAX/MIN are gfortran's: MAX(a, b) keeps a unless b > a or a is NaN
// (and MIN with b < a), the code -O0 emits for the intrinsic.
//
// effectRad's two double-precision statements reduce to FP32 exactly:
//   * re_qc = sngl(3.0/lamc) with lamc a REAL widened to double: a
//     quotient of two binary32 values rounded to binary64 and then to
//     binary32 equals the correctly rounded binary32 quotient (53 >= 2*24+2,
//     so the double rounding is innocuous), i.e. div.rn.f32(3, lamc).
//   * nccps = qccps*6./pi*denr/rho*lammps**3. with qccps = 0 (the driver
//     passes zeros): the product is the signed zero of the REAL factor
//     unless lammps**3 is not finite, which happens exactly when lammps
//     (a REAL widened to double, cubed in double) is infinite or NaN.
//     rkp_nccps carries that case through rather than assuming it away.

__device__ __forceinline__ float rkp_add(float a, float b) {
    float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float rkp_sub(float a, float b) {
    float r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float rkp_mul(float a, float b) {
    float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float rkp_div(float a, float b) {
    float r; asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
#define RK_AD(a, b) rkp_add((a), (b))
#define RK_SU(a, b) rkp_sub((a), (b))
#define RK_MU(a, b) rkp_mul((a), (b))
#define RK_DV(a, b) rkp_div((a), (b))

__device__ __forceinline__ float rkp_max(float a, float b) {
    return (b > a || isnan(a)) ? b : a;
}
__device__ __forceinline__ float rkp_min(float a, float b) {
    return (b < a || isnan(a)) ? b : a;
}
__device__ __forceinline__ float rkp_sqrt(float x) {
    return __fsqrt_rn(x);
}

// effectRad constants (module_ra_effective_radius.F:1-30): pi = 4*atan(1.)
// folded to binary32 (0x40490fdb), denr = rhowater = 1000, dens = rhosnow
// = 100, n0s = 2e6, n0g = 4e6, n0smax = 1e11, nc0 = 3e8, deng = 500,
// alpha = .12; obmr = 1.0/3.0 and n0smax/n0s folded by the compiler.
#define RKP_PI __uint_as_float(0x40490fdbu)
#define RKP_R1 1.e-12f
#define RKP_R2 1.e-6f

struct RkpRe { float qc, qi, qs; };

// rho*max(qi, qmin) -> sqrt(sqrt(temp*temp*temp)) -> ni.
__device__ __forceinline__ float rkp_ni(float qi, float rho) {
    float temp = RK_MU(rho, rkp_max(qi, 0.0f));
    temp = rkp_sqrt(rkp_sqrt(RK_MU(RK_MU(temp, temp), temp)));
    return rkp_min(rkp_max(RK_MU(5.38e7f, temp), 1.e3f), 1.e6f);
}

// nccps(k) = qccps*6./pi*denr/rho(k)*lammps**3. with qccps = 0.
__device__ __forceinline__ float rkp_nccps(float qc, float nc, float rho,
                                           int f_qnc, float cdm2,
                                           float pidnc, float obmr) {
    float rqcmps = rkp_max(RKP_R1, RK_MU(RK_SU(qc, 0.0f), rho));
    float lammps;
    if (f_qnc)
        lammps = RK_MU(RK_MU(2.0f, cdm2),
                       gfk_pow(RK_DV(RK_MU(pidnc, rkp_max(0.0f, nc)), rqcmps),
                               obmr));
    else
        lammps = gfk_pow(RK_DV(RK_MU(pidnc, 3.e8f), rqcmps), obmr);
    float lead = RK_DV(RK_MU(RK_DV(RK_MU(0.0f, 6.0f), RKP_PI), 1000.0f), rho);
    return RK_MU(lead, isfinite(lammps) ? 1.0f : lammps);
}

// One column of effectRad, the driver's x1e6 and clamps.  Two passes:
// the has_q* flags are column-wide, then the per-level radii.
__device__ void rkp_effect_rad(int nz, const float* t, const float* qc,
                               const float* nc, const float* qi,
                               const float* qs, const float* qg,
                               const float* rho, int f_qnc, float cdm2,
                               float* re_qc, float* re_qi, float* re_qs)
{
    const float pidnc = RK_DV(RK_MU(RKP_PI, 1000.0f), 6.0f);
    const float pidn0s = RK_MU(RK_MU(RKP_PI, 100.0f), 2.e6f);
    const float pidn0g = RK_MU(RK_MU(RKP_PI, 500.0f), 4.e6f);
    const float obmr = RK_DV(1.0f, 3.0f);
    bool has_qc = false, has_qi = false, has_qs = false;
    for (int k = 0; k < nz; ++k) {
        float rqc = rkp_max(RKP_R1, RK_MU(qc[k], rho[k]));
        float nccps = rkp_nccps(qc[k], nc[k], rho[k], f_qnc, cdm2, pidnc, obmr);
        float rnc = rkp_max(RKP_R2, RK_MU(RK_AD(nc[k], nccps), rho[k]));
        if (rqc > RKP_R1 && rnc > RKP_R2) has_qc = true;
        float rqi = rkp_max(RKP_R1, RK_MU(qi[k], rho[k]));
        float rni = rkp_max(RKP_R2, RK_MU(rkp_ni(qi[k], rho[k]), rho[k]));
        if (rqi > RKP_R1 && rni > RKP_R2) has_qi = true;
        float rqs = rkp_max(RKP_R1, RK_MU(qs[k], rho[k]));
        float rqg = rkp_max(RKP_R1, RK_MU(qg[k], rho[k]));
        if (rqs > RKP_R1 || rqg > RKP_R1) has_qs = true;
    }
    for (int k = 0; k < nz; ++k) {
        float vqc = 5.0e-6f, vqi = 10.0e-6f, vqs = 25.0e-6f;
        if (has_qc) {
            float rqc = rkp_max(RKP_R1, RK_MU(qc[k], rho[k]));
            float nccps = rkp_nccps(qc[k], nc[k], rho[k], f_qnc, cdm2,
                                    pidnc, obmr);
            float rnc = rkp_max(RKP_R2, RK_MU(RK_AD(nc[k], nccps), rho[k]));
            if (!(rqc <= RKP_R1 || rnc <= RKP_R2)) {
                float lamc = RK_MU(RK_MU(2.0f, cdm2),
                    gfk_pow(RK_DV(RK_MU(pidnc, RK_AD(rkp_max(0.0f, nc[k]), nccps)),
                                  rqc), obmr));
                vqc = rkp_max(2.51e-6f, rkp_min(RK_DV(3.0f, lamc), 50.e-6f));
            }
        }
        if (has_qi) {
            float rqi = rkp_max(RKP_R1, RK_MU(qi[k], rho[k]));
            float ni = rkp_ni(qi[k], rho[k]);
            float rni = rkp_max(RKP_R2, RK_MU(ni, rho[k]));
            if (!(rqi <= RKP_R1 || rni <= RKP_R2)) {
                float diai = RK_MU(11.9f, rkp_sqrt(RK_DV(rqi, ni)));
                vqi = rkp_max(10.01e-6f,
                              rkp_min(RK_MU(RK_MU(0.75f, 0.163f), diai), 125.e-6f));
            }
        }
        if (has_qs) {
            float rqs = rkp_max(RKP_R1, RK_MU(qs[k], rho[k]));
            if (!(rqs <= RKP_R1)) {
                float supcol = RK_SU(273.15f, t[k]);
                float n0sfac = rkp_max(rkp_min(gfk_exp(RK_MU(0.12f, supcol)),
                                               50000.0f), 1.0f);
                float lamdas = rkp_sqrt(rkp_sqrt(
                    RK_DV(RK_MU(pidn0s, n0sfac), rqs)));
                vqs = rkp_max(25.e-6f, rkp_min(RK_MU(0.5f, RK_DV(1.0f, lamdas)),
                                               999.e-6f));
            }
            float vqg = 25.e-6f;
            float rqg = rkp_max(RKP_R1, RK_MU(qg[k], rho[k]));
            if (!(rqg <= RKP_R1)) {
                float lamdag = rkp_sqrt(rkp_sqrt(
                    RK_DV(RK_MU(pidn0g, 4.e6f), rqg)));
                vqg = rkp_max(25.e-6f, rkp_min(RK_MU(1.5f, RK_DV(1.0f, lamdag)),
                                               999.e-6f));
            }
            float qsqg = rkp_max(RKP_R1, RK_AD(qs[k], qg[k]));
            vqs = RK_DV(RK_AD(RK_MU(vqs, rkp_max(0.0f, qs[k])),
                              RK_MU(vqg, rkp_max(0.0f, qg[k]))), qsqg);
        }
        // driver: x1e6 and the 5-1 clamps
        vqc = RK_MU(vqc, 1.e6f);
        vqi = RK_MU(vqi, 1.e6f);
        vqs = RK_MU(vqs, 1.e6f);
        re_qc[k] = rkp_max(2.51f, rkp_min(vqc, 50.0f));
        re_qi[k] = rkp_max(10.01f, rkp_min(vqi, 125.0f));
        re_qs[k] = rkp_max(25.0f, rkp_min(vqs, 999.0f));
    }
}

// The wrapper's o3data integral (identical to the legacy wrapper's;
// kernels/rrtmg_legacy_prep.cu rp_o3).
__device__ float rkp_o3(float p, float p1, const float* ow, const float* ph) {
    float acc = 0.0f;
    for (int j = 0; j < 31; ++j) {
        float d;
        d = RK_SU(p, ph[j]);      float pb1 = d > 0.0f ? d : 0.0f;
        d = RK_SU(p, ph[j + 1]);  float pb2 = d > 0.0f ? d : 0.0f;
        d = RK_SU(p1, ph[j]);     float pt1 = d > 0.0f ? d : 0.0f;
        d = RK_SU(p1, ph[j + 1]); float pt2 = d > 0.0f ? d : 0.0f;
        float w = RK_AD(RK_SU(RK_SU(pb2, pb1), pt2), pt1);
        acc = RK_AD(acc, RK_MU(w, ow[j]));
    }
    return RK_DV(acc, RK_SU(p, p1));
}

// One thread per column.  Inputs (ncol, nz) / (ncol, nz+1) row-major;
// outputs (ncol, nl) / (ncol, nl+1) with nl = nz + 1.  re_diag (3, ncol,
// nz) receives the clamped effective radii before the snow-mass cap (the
// oracle's re_qc/re_qi/re_qs records); null skips it.
extern "C" __global__ void rkp_prep(
    int ncol, int nz, int inflg, int iceflg, int o3input, int f_qnc,
    float cdm2,
    const float* __restrict__ p3d, const float* __restrict__ p8w,
    const float* __restrict__ t3d, const float* __restrict__ t8w,
    const float* __restrict__ rho, const float* __restrict__ qv,
    const float* __restrict__ qc, const float* __restrict__ qi,
    const float* __restrict__ qs, const float* __restrict__ qg,
    const float* __restrict__ nc, const float* __restrict__ cldfra,
    const float* __restrict__ o33d,
    const float* __restrict__ o3wrk, const float* __restrict__ ppwrkh,
    float* play, float* plev, float* tlay, float* tlev, float* h2ovmr,
    float* o3vmr, float* cldfrac, float* clwp, float* ciwp, float* cswp,
    float* rel, float* rei, float* res, float* re_diag)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const int nl = nz + 1;
    const size_t a = (size_t)col * nz, a1 = (size_t)col * (nz + 1);
    const size_t b = (size_t)col * nl, b1 = (size_t)col * (nl + 1);
    const float gravdvd = RK_DV(1000.0f, 9.81f);
    // effectRad into the layer radius slots (rel/rei/res, first nz).
    rkp_effect_rad(nz, t3d + a, qc + a, nc + a, qi + a, qs + a, qg + a,
                   rho + a, f_qnc, cdm2, rel + b, rei + b, res + b);
    if (re_diag != nullptr) {
        for (int k = 0; k < nz; ++k) {
            re_diag[a + k] = rel[b + k];
            re_diag[(size_t)ncol * nz + a + k] = rei[b + k];
            re_diag[2 * (size_t)ncol * nz + a + k] = res[b + k];
        }
    }
    for (int k = 0; k < nz; ++k) {
        float cf = cldfra[a + k];
        float qsum = RK_AD(RK_AD(qi[a + k], qc[a + k]), qs[a + k]);
        float pdel = RK_SU(p8w[a1 + k], p8w[a1 + k + 1]);
        float cden = rkp_max(0.01f, cf);
        bool thin = cf > 0.0f && qsum < 1.e-9f;
        float gice = RK_MU(RK_MU(RK_AD(qi[a + k], qs[a + k]), pdel), gravdvd);
        float gliq = RK_MU(RK_MU(qc[a + k], pdel), gravdvd);
        float cice = RK_DV(gice, cden);
        float cliq = RK_DV(gliq, cden);
        if (thin) cliq = RK_MU(RK_MU(1.e-9f, pdel), gravdvd);
        if (iceflg >= 4) {
            cice = RK_DV(RK_MU(RK_MU(qi[a + k], pdel), gravdvd), cden);
            if (thin) cice = RK_MU(RK_MU(1.e-9f, pdel), gravdvd);
        }
        float csno = 0.0f;
        float rs = res[b + k];
        if (iceflg == 5) {
            float smf = 1.0f;
            if (rs > 130.0f) {
                smf = gfk_pow(RK_DV(130.0f, rs), 2.0f);
                rs = 130.0f;
            }
            float gsno = RK_MU(RK_MU(RK_MU(RK_AD(qs[a + k], qg[a + k]), smf),
                                     pdel), gravdvd);
            if (thin) gsno = 0.0f;
            csno = RK_DV(gsno, cden);
        }
        float ri = rei[b + k];
        if (iceflg == 3) ri = rkp_min(140.0f, RK_MU(ri, 1.0315f));
        clwp[b + k] = cliq;
        ciwp[b + k] = cice;
        rei[b + k] = ri;
        if (inflg == 5) {
            cswp[b + k] = csno;
            res[b + k] = rs;
        } else {
            cswp[b + k] = 0.0f;
            res[b + k] = 10.0f;
        }
        cldfrac[b + k] = cf;
        play[b + k] = RK_MU(p3d[a + k], 1.e-2f);
        tlay[b + k] = t3d[a + k];
        h2ovmr[b + k] = RK_MU(rkp_max(rkp_max(0.0f, qv[a + k]), 3.0e-6f),
                              1.607793f);
    }
    for (int k = 0; k <= nz; ++k) {
        plev[b1 + k] = RK_MU(p8w[a1 + k], 1.e-2f);
        tlev[b1 + k] = t8w[a1 + k];
    }
    // the layer above the model top
    play[b + nz] = RK_MU(0.5f, plev[b1 + nz]);
    tlay[b + nz] = RK_AD(tlev[b1 + nz], 0.0f);
    plev[b1 + nz + 1] = 1.0e-5f;
    tlev[b1 + nz + 1] = RK_AD(tlev[b1 + nz], 0.0f);
    h2ovmr[b + nz] = h2ovmr[b + nz - 1];
    clwp[b + nz] = 0.0f; ciwp[b + nz] = 0.0f; cswp[b + nz] = 0.0f;
    rel[b + nz] = 10.0f; rei[b + nz] = 10.0f; res[b + nz] = 10.0f;
    cldfrac[b + nz] = 0.0f;
    // inirad / o3data over kts..kte+1, then o3input = 2's model levels
    const float amdo = 0.603461f;
    for (int k = 0; k <= nz; ++k)
        o3vmr[b + k] = RK_MU(rkp_o3(plev[b1 + k], plev[b1 + k + 1], o3wrk, ppwrkh),
                             amdo);
    if (o3input == 2) {
        float m_top = rkp_o3(plev[b1 + nz - 1], plev[b1 + nz], o3wrk, ppwrkh);
        float m_up = rkp_o3(plev[b1 + nz], plev[b1 + nz + 1], o3wrk, ppwrkh);
        for (int k = 0; k < nz; ++k) o3vmr[b + k] = o33d[a + k];
        float v = RK_AD(RK_SU(o33d[a + nz - 1], RK_MU(m_top, amdo)),
                        RK_MU(m_up, amdo));
        o3vmr[b + nz] = v <= 0.0f ? RK_MU(m_up, amdo) : v;
    }
}

// generate_stochastic_redu, icld = 2, irng = 0 (kissvec): one thread per
// (column, subcolumn 1..28).  The column's stream is seeded from its bottom
// four pmid = play*1e2 words, advanced change_seed times, then drawn 28 x nl
// in (subcolumn outer, level inner) order; this thread skips to its own
// subcolumn's draws.  The walk runs from the top level down, keeping the
// level-above value where it exceeds that level's clear fraction and
// otherwise scaling this level's draw by it; the 28 walked subcolumns are
// replicated five times to the 140 g-points, and a subcolumn is cloudy
// where cdf >= 1 - cldf.  cdf is (ncol, 28, nl) scratch; outputs are
// (ncol, nl, 140).  err bit 1: pmid(1) < pmid(2) (the Fortran STOP).
extern "C" __global__ void rkp_mcica(
    int ncol, int nl, int change_seed,
    const float* __restrict__ play, const float* __restrict__ cldfrac,
    const float* __restrict__ ciwp, const float* __restrict__ clwp,
    const float* __restrict__ cswp, float* cdf,
    float* cldfmcl, float* ciwpmcl, float* clwpmcl, float* cswpmcl,
    int* err)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)ncol * 28) return;
    int col = (int)(t / 28);
    int isub = (int)(t % 28);
    const float* pc = play + (size_t)col * nl;
    const float* cf = cldfrac + (size_t)col * nl;
    if (isub == 0 && RK_MU(pc[0], 1.0e2f) < RK_MU(pc[1], 1.0e2f))
        atomicOr(err, 1);
    unsigned int s1 = rmcw_seed_from_play(pc[0]);
    unsigned int s2 = rmcw_seed_from_play(pc[1]);
    unsigned int s3 = rmcw_seed_from_play(pc[2]);
    unsigned int s4 = rmcw_seed_from_play(pc[3]);
    for (int i = 0; i < change_seed; ++i) rmcw_kiss_advance(s1, s2, s3, s4);
    for (long long i = 0; i < (long long)isub * nl; ++i)
        rmcw_kiss_advance(s1, s2, s3, s4);
    float* c = cdf + ((size_t)col * 28 + isub) * nl;
    for (int l = 0; l < nl; ++l) c[l] = rmcw_kiss_draw(s1, s2, s3, s4);
    for (int l = nl - 2; l >= 0; --l) {
        float cfa = cf[l + 1];
        cfa = (cfa < 1.0e-20f) ? 0.0f : cfa;
        float clear = RK_SU(1.0f, cfa);
        c[l] = (c[l + 1] > clear) ? c[l + 1] : RK_MU(c[l], clear);
    }
    for (int l = 0; l < nl; ++l) {
        float cfl = cf[l];
        cfl = (cfl < 1.0e-20f) ? 0.0f : cfl;
        bool cloudy = c[l] >= RK_SU(1.0f, cfl);
        size_t s = (size_t)col * nl + l;
        float vi = cloudy ? ciwp[s] : 0.0f;
        float vl = cloudy ? clwp[s] : 0.0f;
        float vs = cloudy ? cswp[s] : 0.0f;
        float vc = cloudy ? 1.0f : 0.0f;
        for (int r = 0; r < 5; ++r) {
            size_t o = s * 140 + isub + 28 * r;
            cldfmcl[o] = vc; ciwpmcl[o] = vi; clwpmcl[o] = vl; cswpmcl[o] = vs;
        }
    }
}

// Driver outputs (no Exner division: rad_rrtmg_driver writes hr/86400 to
// RTHRATEN and swhr/86400 to RTHRATENSW).  One thread per (column, level);
// results land in (ncol, nz) row-major slabs at column col0 + c (LW) or
// sel[c] (SW, the day columns).
extern "C" __global__ void rkp_lw_out(
    int nc, int nz, int nl, long long col0,
    const float* __restrict__ hr, const float* __restrict__ uflx,
    const float* __restrict__ dflx, float* rthratenlw, float* glw, float* olr)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)nc * nz) return;
    int c = (int)(t / nz), k = (int)(t % nz);
    long long col = col0 + c;
    rthratenlw[(size_t)col * nz + k] = RK_DV(hr[(size_t)c * nl + k], 86400.0f);
    if (k == 0) {
        glw[col] = dflx[(size_t)c * (nl + 1)];
        olr[col] = uflx[(size_t)c * (nl + 1) + nz + 1];
    }
}

extern "C" __global__ void rkp_sw_out(
    int nc, int nz, int nl, const long long* __restrict__ sel,
    const float* __restrict__ swhr, const float* __restrict__ swuflx,
    const float* __restrict__ swdflx, float* rthratensw, float* gsw)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)nc * nz) return;
    int c = (int)(t / nz), k = (int)(t % nz);
    long long col = sel[c];
    rthratensw[(size_t)col * nz + k] = RK_DV(swhr[(size_t)c * nl + k], 86400.0f);
    if (k == 0)
        gsw[col] = RK_SU(swdflx[(size_t)c * (nl + 1)], swuflx[(size_t)c * (nl + 1)]);
}

// RTHRATEN = RTHRATENLW + RTHRATENSW (driver :2702-2710), the published
// RTHRATENSW = RTHRATEN - RTHRATENLW, and SWDOWN = GSW/(1-ALBEDO).
extern "C" __global__ void rkp_driver(
    long long n, long long ncol, const float* __restrict__ lw, float* sw,
    float* total, const float* __restrict__ gsw,
    const float* __restrict__ albedo, float* swdown)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) {
        float r = RK_AD(lw[t], sw[t]);
        total[t] = r;
        sw[t] = RK_SU(r, lw[t]);
    }
    if (t < ncol) swdown[t] = RK_DV(gsw[t], RK_SU(1.0f, albedo[t]));
}

// Probe: subnormal survival, gfk_pow/gfk_exp, gfortran MAX on NaN.
extern "C" __global__ void rkp_probe(const float* x, float* y)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    y[0] = RK_MU(x[0], x[1]);
    y[1] = gfk_pow(x[2], x[3]);
    y[2] = gfk_exp(x[4]);
    y[3] = rkp_max(x[5], 1.0f);
}
