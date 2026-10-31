/* Deng shallow cumulus (WRF shcu_physics = 5, WRF-Solar), one CUDA thread
 * per column.
 *
 * AUTHORITY: WRF v4.7.1 phys/module_shcu_deng.F (sha256 d03fd435...9645, LF)
 * driven through phys/module_shallowcu_driver.F (068cf22b...271a), graded
 * word for word by the column-tile oracle in tools/shcu_deng_wrf471_oracle:
 * the unmodified Fortran at gfortran -O0 -ffp-contract=off, every output of
 * shallowcu_driver's Deng arm over consecutive steps, 0 ULP.
 *
 * This file is a line-for-line port of deng_shcu_driver
 * (module_shcu_deng.F:17-426, the per-column part: the W time average, the
 * RH-based radiation cloud, the call, the tendency/MAVAIL/AINCKFSA
 * book-keeping and the NBC leapfrog), deng_shcu (:429-3571) and the helpers
 * it calls -- CONDLOAD (:3768), DTFRZNEW (:3833), ENVIRTHT (:3951), PROF5
 * (:4049), TPDDBG (:4165), TPMIX (:4265), GAMMA (:4823), MINIM (:5080) and
 * MAXIM (:5094) -- plus shallowcu_driver's own two Deng lines: the TKE
 * source by bl_pbl_physics (module_shallowcu_driver.F:605-630) and the
 * closing RAINSHV = PRATESH*DT (:686-691).
 *
 * Every phase of deng_shcu_driver is column-local, so running the whole
 * driver for one column in one thread reproduces the tile loops exactly.
 *
 * How the transcription is held to the Fortran (the kf.cu rules):
 *   - Arrays keep the Fortran names and the Fortran 1-based index.
 *   - Every expression keeps Fortran's left-to-right evaluation order; the
 *     unit compiles with --fmad=false --ftz=false on its own route
 *     (woof/core/shcu_deng.py), so nothing is contracted or flushed.
 *   - EXP, ALOG and REAL**REAL call WOOF's float32 libm (gfk_exp, gfk_log,
 *     gfk_pow from glibc_flt32.cuh), the very functions the oracle links in
 *     place of the host library (tools/shcu_deng_wrf471_oracle/libm_shim.cpp).
 *     SQRT and division are IEEE on both sides.  NINT rounds half away
 *     (roundf), IFIX/INT truncate.  MAX/MIN of REALs are fmaxf/fminf: the
 *     oracle's gfortran emits fmaxnm/fminnm, which order -0 below +0 and
 *     ignore a NaN operand, as fmaxf/fminf do.
 *   - GOTOs become structured control flow with the same targets; no
 *     variable the Fortran carries across a label is reset here.
 *   - State the Fortran mutates in place and reads again later in the same
 *     call -- KLCL re-set by the post-convective LCL search (:2367-2370) and
 *     read by the downdraft rescale (:2576), ROCPQ read by the NBC radiative
 *     diffusion coefficient (:3018) from whichever branch last set it, K
 *     re-set by the PBL search (:967-979) -- is carried the same way.
 *
 * Where WRF itself stops (STOP 907 and 901-904, the ice-settling NSTEP
 * guard) or reads memory it never wrote (a mass-imbalance exit before TG is
 * formed; a post-convective LCL at the lowest level, K = 0), the column
 * records a status bit (DENG_STATUS_*) and the host refuses the step: WRF
 * has no defined answer there.  No fixture column reaches any of them.
 */

/* gfk_exp/gfk_log/gfk_pow: glibc_flt32.cuh, prepended by module_source
 * (woof/core/kernels/__init__.py _EXTRA_HEADERS["shcu_deng"]). */

#ifndef DENG_KMAX
#define DENG_KMAX 256
#endif

#define DENG_STATUS_STOP907 1
#define DENG_STATUS_STOP90X 2
#define DENG_STATUS_SETTLING 4
#define DENG_STATUS_MASS_IMBALANCE 8
#define DENG_STATUS_LCL_AT_SURFACE 16
#define DENG_STATUS_GAMMA 32
#define DENG_STATUS_NO_500HPA 64

/* ==========================================================================
 * The per-thread column workspace (kf.cu's layout): element k of slot s for
 * lane t of a warp's region is region + (s * kp + k - 1) * 32 + t, with
 * kp = nz + 1 (the longest Fortran extent, kts:kte+1).  Slot ids are
 * literal; tests/test_shcu_deng_wrf471_oracle.py reads them out of this
 * file and fails on a duplicate or an id past DENGWS_SLOTS.
 * ======================================================================== */
#define DENGWS_SLOTS 88
#define DENGWS_LANES 32

struct DCol {
    float* p;
    __device__ __forceinline__ float& operator()(int k) const {
        return p[(size_t)(k - 1) * (size_t)DENGWS_LANES];
    }
};
/* A read-only (nz, ny, nx) input column with a 1-based level index. */
struct DIn {
    const float* __restrict__ p;
    int ncol;
    int column;
    __device__ __forceinline__ float operator()(int k) const {
        return p[(size_t)(k - 1) * (size_t)ncol + column];
    }
};
/* A read-write (nz, ny, nx) state column with a 1-based level index. */
struct DOut {
    float* p;
    int ncol;
    int column;
    __device__ __forceinline__ float& operator()(int k) const {
        return p[(size_t)(k - 1) * (size_t)ncol + column];
    }
};

#define DWS(base, idx, kp) \
    (DCol{(base) + (size_t)(idx) * (size_t)(kp) * (size_t)DENGWS_LANES})
#define DENGWS_LANE_BASE(ws, kp) \
    ((ws) + ((size_t)blockIdx.x * (blockDim.x / DENGWS_LANES) \
             + threadIdx.x / DENGWS_LANES) * DENGWS_SLOTS * (kp) \
             * DENGWS_LANES + threadIdx.x % DENGWS_LANES)

/* share/module_model_constants.F (v4.7.1), the values
 * module_shallowcu_driver passes deng_shcu_driver. */
#define DENG_G 9.81f
#define DENG_R 287.0f
#define DENG_CP 1004.5f   /* 7.*r_d/2. */
#define DENG_XLV 2.5E6f
#define DENG_XLS 2.85E6f
#define DENG_XLV0 3.15E6f
#define DENG_XLV1 2370.0f
#define DENG_XLS0 2.905E6f
#define DENG_XLS1 259.532f
#define DENG_SVP1 0.6112f
#define DENG_SVP2 17.67f
#define DENG_SVP3 29.65f
#define DENG_SVPT0 273.15f
/* module_shcu_deng.F:7 */
#define DENG_TO 263.15f

__device__ __forceinline__ int deng_nint(float x) { return (int)roundf(x); }

/* 1.E3*SVP1*EXP(SVP2*(T-SVPT0)/(T-SVP3)), the scheme's liquid es. */
__device__ __forceinline__ float deng_esliq(float t) {
    return (1.E3f * DENG_SVP1)
        * gfk_exp((DENG_SVP2 * (t - DENG_SVPT0)) / (t - DENG_SVP3));
}
/* 611.0*EXP(22.514-6.15E3/T), the scheme's ice es. */
__device__ __forceinline__ float deng_esice(float t) {
    return 611.0f * gfk_exp(22.514f - 6.15E3f / t);
}
/* X**(0.2854*(1.-0.28*Q)) */
__device__ __forceinline__ float deng_exner_pow(float x, float q) {
    return gfk_pow(x, 0.2854f * (1.0f - 0.28f * q));
}
/* EXP((3374.6525/T-2.5403)*Q*(1.+0.81*Q)) */
__device__ __forceinline__ float deng_bolton(float t, float q) {
    return gfk_exp(((3374.6525f / t - 2.5403f) * q) * (1.0f + 0.81f * q));
}

/* -------------------------------------------------------------- CONDLOAD */
/* module_shcu_deng.F:3768-3831 */
__device__ __forceinline__ void deng_condload(
        float* qliq, float* qice, float* wtw, float dz, float boterm,
        float enterm, float rate, float* qnewlq, float* qnewic,
        float* qlqout, float* qicout, float g) {
    if (fabsf(*wtw) < 1.E-4f) *wtw = 1.E-4f;
    float qtot = *qliq + *qice;
    float qnew = *qnewlq + *qnewic;
    float qest = 0.5f * (qtot + qnew);
    float g1 = ((*wtw + boterm) - enterm) - (((2.0f * g) * dz) * qest) / 1.5f;
    if (g1 < 0.0f) g1 = 0.0f;
    float wavg = 0.5f * (sqrtf(*wtw) + sqrtf(g1));
    float conv = (rate * dz) / wavg;
    float ratio3 = *qnewlq / (qnew + 1.E-8f);
    qtot = qtot + 0.6f * qnew;
    float oldq = qtot;
    float ratio4 = (0.6f * *qnewlq + *qliq) / (qtot + 1.E-8f);
    qtot = qtot * gfk_exp(-conv);
    float dq = oldq - qtot;
    *qlqout = ratio4 * dq;
    *qicout = (1.0f - ratio4) * dq;
    float pptdrg = 0.5f * ((oldq + qtot) - 0.2f * qnew);
    *wtw = ((*wtw + boterm) - enterm) - (((2.0f * g) * dz) * pptdrg) / 1.5f;
    if (fabsf(*wtw) < 1.E-4f) *wtw = 1.E-4f;
    *qliq = ratio4 * qtot + (ratio3 * 0.4f) * qnew;
    *qice = (1.0f - ratio4) * qtot + ((1.0f - ratio3) * 0.4f) * qnew;
    *qnewlq = 0.0f;
    *qnewic = 0.0f;
}

/* -------------------------------------------------------------- DTFRZNEW */
/* module_shcu_deng.F:3833-3947 */
__device__ __forceinline__ void deng_dtfrznew(
        float* tu, float p, float* thteu, float* qvap, float* qliq,
        float* qice, float* ratio2, float* qnwfrz, float* rl, float* frc1,
        float effq, int* iflag) {
    const float c5 = 1.0723E-3f;
    float qlqfrz = *qliq * effq;
    float qnew = (*qnwfrz * effq) * 0.5f;
    float esliq = deng_esliq(*tu);
    float esice = deng_esice(*tu);
    float rlc = 2.5E6f - 2369.276f * (*tu - 273.16f);
    float rls = 2833922.0f - 259.532f * (*tu - 273.16f);
    float rlf = rls - rlc;
    float cp = 1005.7f * (1.0f + 0.89f * *qvap);
    float a = 6.15E+3f / (*tu * *tu);
    float b = (rls * 0.622f) / p;
    float c = ((a * b) * esice) / cp;
    float dqvap = (b * (esliq - esice)) / (rls + rls * c)
                  - (rlf * (qlqfrz + qnew)) / (rls + rls / c);
    float dtfrz = (rlf * (qlqfrz + qnew) + b * (esliq - esice))
                  / (cp + (a * b) * esice);
    float tu1 = *tu;
    float qvap1 = *qvap;
    *tu = *tu + *frc1 * dtfrz;
    *qvap = *qvap - *frc1 * dqvap;
    float es = (*qvap * p) / (0.622f + *qvap);
    esliq = deng_esliq(*tu);
    esice = deng_esice(*tu);
    *ratio2 = (esliq - es) / (esliq - esice);
    if (*iflag > 0 && *ratio2 < 1.0f) {
        *frc1 = *frc1 + (1.0f - *ratio2);
        *tu = tu1 + *frc1 * dtfrz;
        *qvap = qvap1 - *frc1 * dqvap;
        *ratio2 = 1.0f;
        *iflag = 1;
    } else if (*ratio2 > 1.0f) {
        *frc1 = *frc1 - (*ratio2 - 1.0f);
        *frc1 = fmaxf(0.0f, *frc1);
        *tu = tu1 + *frc1 * dtfrz;
        *qvap = qvap1 - *frc1 * dqvap;
        *ratio2 = 1.0f;
        *iflag = 1;
    }
    /* label 20 */
    rlc = DENG_XLV0 - DENG_XLV1 * *tu;
    rls = DENG_XLS0 - DENG_XLS1 * *tu;
    *rl = *ratio2 * rls + (1.0f - *ratio2) * rlc;
    float pi = deng_exner_pow(1.E5f / p, *qvap);
    *thteu = (*tu * pi)
             * gfk_exp((((*rl * *qvap) * c5) / *tu) * (1.0f + 0.81f * *qvap));
    if (*iflag == 1) {
        *qice = (*qice + *frc1 * dqvap) + *qliq;
        *qliq = 0.0f;
    } else {
        *qice = *qice + *frc1 * (dqvap + qlqfrz);
        *qliq = *qliq - *frc1 * qlqfrz;
    }
    *qnwfrz = 0.0f;
}

/* -------------------------------------------------------------- ENVIRTHT */
/* module_shcu_deng.F:3951-4007 */
__device__ __forceinline__ float deng_envirtht(float p1, float t1, float q1,
                                               float r1, float rl) {
    const float t00 = 273.16f, p00 = 1.E5f, c1 = 3374.6525f, c2 = 2.5403f;
    const float c3 = 3114.834f, c4 = 0.278296f, c5 = 1.0723E-3f;
    float ee = fmaxf((q1 * p1) / (0.622f + q1), 1.e-9f);
    if (r1 < 1.E-6f) {
        float tlog = gfk_log((ee / DENG_SVP1) / 1000.0f);
        float tdpt = (DENG_SVP2 * DENG_SVPT0 - DENG_SVP3 * tlog)
                     / (DENG_SVP2 - tlog);
        float tsat = tdpt - ((0.212f + 1.571E-3f * (tdpt - t00))
                             - 4.36E-4f * (t1 - t00)) * (t1 - tdpt);
        float tht = t1 * deng_exner_pow(p00 / p1, q1);
        return tht * gfk_exp(((c1 / tsat - c2) * q1) * (1.0f + 0.81f * q1));
    } else if (fabsf(r1 - 1.0f) < 1.E-6f) {
        float tlog = gfk_log(ee / 611.0f);
        float tfpt = 6150.0f / (22.514f - tlog);
        float tht = t1 * deng_exner_pow(p00 / p1, q1);
        float tsat = tfpt - ((0.182f + 1.13E-3f * (tfpt - t00))
                             - 3.58E-4f * (t1 - t00)) * (t1 - tfpt);
        return tht * gfk_exp(((c3 / tsat - c4) * q1) * (1.0f + 0.81f * q1));
    }
    float tlog = gfk_log((ee / DENG_SVP1) / 1000.0f);
    float tdpt = (DENG_SVP2 * DENG_SVPT0 - DENG_SVP3 * tlog)
                 / (DENG_SVP2 - tlog);
    float tlogic = gfk_log(ee / 611.0f);
    float tfpt = 6150.0f / (22.514f - tlogic);
    float tht = t1 * deng_exner_pow(p00 / p1, q1);
    float tsatlq = tdpt - ((0.212f + 1.571E-3f * (tdpt - t00))
                           - 4.36E-4f * (t1 - t00)) * (t1 - tdpt);
    float tsatic = tfpt - ((0.182f + 1.13E-3f * (tfpt - t00))
                           - 3.58E-4f * (t1 - t00)) * (t1 - tfpt);
    float tsat = r1 * tsatic + (1.0f - r1) * tsatlq;
    return tht * gfk_exp((((rl * q1) * c5) / tsat) * (1.0f + 0.81f * q1));
}

/* ----------------------------------------------------------------- PROF5 */
/* module_shcu_deng.F:4049-4088.  E45 = EXP(-4.5) is folded by gfortran at
 * compile time; tools/kf_wrf461_oracle/e45_probe.F90 measured the folded
 * word (the same expression in module_cu_kfeta.F): 0x3c360282. */
#define DENG_PROF5_E45 __uint_as_float(0x3c360282u)
__device__ __forceinline__ void deng_prof5(float eq, float* ee, float* ud) {
    const float sqrt2p = 2.506628f, a1 = 0.4361836f, a2 = -0.1201676f;
    const float a3 = 0.9372980f, p = 0.33267f, sigma = 0.166666667f;
    const float fe = 0.202765151f;
    float y = 6.0f * eq - 3.0f;
    float ey = gfk_exp((y * y) / (-2.0f));
    float e45 = DENG_PROF5_E45;
    float t2 = 1.0f / (1.0f + p * fabsf(y));
    float t1 = 0.500498f;
    float c1 = (a1 * t1 + (a2 * t1) * t1) + ((a3 * t1) * t1) * t1;
    float c2 = (a1 * t2 + (a2 * t2) * t2) + ((a3 * t2) * t2) * t2;
    if (y >= 0.0f) {
        *ee = sigma * (0.5f * ((sqrt2p - e45 * c1) - ey * c2)
                       + sigma * (e45 - ey))
              - ((e45 * eq) * eq) / 2.0f;
        *ud = sigma * (0.5f * (ey * c2 - e45 * c1) + sigma * (e45 - ey))
              - e45 * ((0.5f + (eq * eq) / 2.0f) - eq);
    } else {
        *ee = sigma * (0.5f * (ey * c2 - e45 * c1) + sigma * (e45 - ey))
              - ((e45 * eq) * eq) / 2.0f;
        *ud = sigma * (0.5f * ((sqrt2p - e45 * c1) - ey * c2)
                       + sigma * (e45 - ey))
              - e45 * ((0.5f + (eq * eq) / 2.0f) - eq);
    }
    *ee = *ee / fe;
    *ud = *ud / fe;
}

/* ---------------------------------------------------------------- TPDDBG */
/* module_shcu_deng.F:4165-4263.  deng_shcu calls it with RH = 1.0 only, so
 * the RH < 1 arm after label 50 is never taken (RH.EQ.1. -> GOTO 110); RS
 * (INOUT) leaves with the last iterate's saturation mixing ratio. */
__device__ __forceinline__ float deng_tpddbg(float p, float thted, float tgs,
                                             float* rs) {
    float es = deng_esliq(tgs);
    *rs = (0.622f * es) / (p - es);
    float pi = deng_exner_pow(1.E5f / p, *rs);
    float thtgs = (tgs * pi) * deng_bolton(tgs, *rs);
    float f0 = thtgs - thted;
    float t2 = tgs - 0.5f * f0;
    float t0 = tgs;
    float t1 = t0;
    float f1 = f0;
    int itcnt = 0;
    for (;;) {
        es = deng_esliq(t2);
        *rs = (0.622f * es) / (p - es);
        pi = deng_exner_pow(1.E5f / p, *rs);
        thtgs = (t2 * pi) * deng_bolton(t2, *rs);
        float f2 = thtgs - thted;
        if ((f1 * f2) < 0.0f) {
            f0 = f1;
            t0 = t1;
        }
        f1 = f2;
        t1 = t2;
        if (fabsf(f1) < 0.05f) break;
        itcnt = itcnt + 1;
        if (itcnt > 10) break;
        float dt = (f1 * (t1 - t0)) / (f1 - f0);
        if (fabsf(dt) < fabsf((t1 - t0) / 2.0f)) dt = (t1 - t0) / 2.0f;
        t2 = t1 - dt;
    }
    return t1;
}

/* ----------------------------------------------------------------- TPMIX */
/* module_shcu_deng.F:4265-4427 */
__device__ __forceinline__ void deng_tpmix_thtgs(
        float p, float t, float ratio2, float rl, float* qs, float* thtgs) {
    const float c5 = 1.0723E-3f;
    float es;
    if (ratio2 < 1.E-6f) {
        es = deng_esliq(t);
        *qs = (0.622f * es) / (p - es);
        float pi = deng_exner_pow(1.E5f / p, *qs);
        *thtgs = (t * pi) * deng_bolton(t, *qs);
    } else if (fabsf(ratio2 - 1.0f) < 1.E-6f) {
        es = deng_esice(t);
        *qs = (0.622f * es) / (p - es);
        float pi = deng_exner_pow(1.E5f / p, *qs);
        *thtgs = (t * pi)
            * gfk_exp(((3114.834f / t - 0.278296f) * *qs) * (1.0f + 0.81f * *qs));
    } else {
        float esliq = deng_esliq(t);
        float esice = deng_esice(t);
        es = (1.0f - ratio2) * esliq + ratio2 * esice;
        *qs = (0.622f * es) / (p - es);
        float pi = deng_exner_pow(1.E5f / p, *qs);
        *thtgs = (t * pi)
            * gfk_exp((((rl * *qs) * c5) / t) * (1.0f + 0.81f * *qs));
    }
}

__device__ __forceinline__ void deng_tpmix(
        float p, float thtu, float* tu, float* qu, float* qliq, float* qice,
        float* qnewlq, float* qnewic, float ratio2, float rl) {
    float qs, thtgs;
    deng_tpmix_thtgs(p, *tu, ratio2, rl, &qs, &thtgs);
    float f0 = thtgs - thtu;
    float t1 = *tu - 0.5f * f0;
    float t0 = *tu;
    int itcnt = 0;
    for (;;) {
        deng_tpmix_thtgs(p, t1, ratio2, rl, &qs, &thtgs);
        float f1 = thtgs - thtu;
        if (fabsf(f1) < 0.01f) break;
        itcnt = itcnt + 1;
        if (itcnt > 10) break;
        float dt = (f1 * (t1 - t0)) / (f1 - f0);
        t0 = t1;
        f0 = f1;
        t1 = t1 - dt;
    }
    /* label 50 */
    float qnew;
    if (qs <= *qu) {
        qnew = *qu - qs;
        *qu = qs;
    } else {
        qnew = 0.0f;
        float dq = qs - *qu;
        float qtot = *qliq + *qice;
        if (qtot >= dq) {
            float dqice = 0.0f;
            float dqliq = 0.0f;
            *qliq = *qliq - (1.0f - ratio2) * dq;
            if (*qliq < 0.0f) {
                dqice = 0.0f - *qliq;
                *qliq = 0.0f;
            }
            *qice = (*qice - ratio2 * dq) + dqice;
            if (*qice < 0.0f) {
                dqliq = 0.0f - *qice;
                *qice = 0.0f;
            }
            *qliq = *qliq + dqliq;
            *qu = qs;
        } else {
            float rll;
            if (ratio2 < 1.E-6f) {
                rll = DENG_XLV0 - DENG_XLV1 * t1;
            } else if (fabsf(ratio2 - 1.0f) < 1.E-6f) {
                rll = DENG_XLS0 - DENG_XLS1 * t1;
            } else {
                rll = rl;
            }
            float cp = 1005.7f * (1.0f + 0.89f * *qu);
            if (qtot < 1.E-10f) {
                t1 = t1 + (rll * (dq / (1.0f + dq))) / cp;
            } else {
                t1 = t1 + (rll * ((dq - qtot) / ((1.0f + dq) - qtot))) / cp;
                *qu = *qu + qtot;
                *qliq = 0.0f;
                *qice = 0.0f;
            }
        }
    }
    /* label 96 */
    *tu = t1;
    *qnewlq = (1.0f - ratio2) * qnew;
    *qnewic = ratio2 * qnew;
}

/* ----------------------------------------------------------------- GAMMA */
/* module_shcu_deng.F:4823-5078, the module's own single-precision Cody
 * gamma (it hides the intrinsic).  deng_shcu calls it at 3.+BVTS and
 * 3.+BVT only; the negative-argument arm (SIN) is refused by status. */
__device__ float deng_gamma(float x, int* status) {
    const float P[8] = {-1.71618513886549492533811E+0f,
                        2.47656508055759199108314E+1f,
                        -3.79804256470945635097577E+2f,
                        6.29331155312818442661052E+2f,
                        8.66966202790413211295064E+2f,
                        -3.14512729688483675254357E+4f,
                        -3.61444134186911729807069E+4f,
                        6.64561438202405440627855E+4f};
    const float Q[8] = {-3.08402300119738975254353E+1f,
                        3.15350626979604161529144E+2f,
                        -1.01515636749021914166146E+3f,
                        -3.10777167157231109440444E+3f,
                        2.25381184209801510330112E+4f,
                        4.75584627752788110767815E+3f,
                        -1.34659959864969306392456E+5f,
                        -1.15132259675553483497211E+5f};
    const float C[7] = {-1.910444077728E-03f, 8.4171387781295E-04f,
                        -5.952379913043012E-04f, 7.93650793500350248E-04f,
                        -2.777777777777681622553E-03f,
                        8.333333333333333331554247E-02f, 5.7083835261E-03f};
    const float one = 1.0E0f, half = 0.5E0f, twelve = 12.0E0f, zero = 0.0E0f;
    const float sqrtpi = 0.9189385332046727417803297E0f;
    const float xbig = 35.040E0f, xminin = 1.18E-38f, eps = 1.19E-7f;
    const float xinf = 3.4E38f;
    float res, y = x, y1, z;
    int n = 0;
    if (y <= zero) {
        *status |= DENG_STATUS_GAMMA;
        return xinf;
    }
    if (y < eps) {
        if (y >= xminin) {
            res = one / y;
        } else {
            return xinf;
        }
    } else if (y < twelve) {
        y1 = y;
        if (y < one) {
            z = y;
            y = y + one;
        } else {
            n = (int)y - 1;
            y = y - (float)n;
            z = y - one;
        }
        float xnum = zero, xden = one;
        for (int i = 0; i < 8; ++i) {
            xnum = (xnum + P[i]) * z;
            xden = xden * z + Q[i];
        }
        res = xnum / xden + one;
        if (y1 < y) {
            res = res / y1;
        } else if (y1 > y) {
            for (int i = 1; i <= n; ++i) {
                res = res * y;
                y = y + one;
            }
        }
    } else {
        if (y <= xbig) {
            float ysq = y * y;
            float sum = C[6];
            for (int i = 0; i < 6; ++i) sum = sum / ysq + C[i];
            sum = (sum / y - y) + sqrtpi;
            sum = sum + (y - half) * gfk_log(y);
            res = gfk_exp(sum);
        } else {
            return xinf;
        }
    }
    return res;
}

/* ===================================================== deng_shcu (1 column)
 * module_shcu_deng.F:429-3571.  Scalars arrive as deng_shcu_driver passes
 * them; the column arrays are the driver's 1-D copies (U1D, ..., read in
 * place from the global inputs where the copy is exact) and the 3-D state
 * (CLDAREAB, CLDLIQB, WUB) at this column. */
struct DengColumn {
    /* 1-D inputs (U0, V0, T0, QR0, RH0, P0, RHOE, DZQ, W0AVG0, TKE0, KTH0,
     * BBLS0, TEN_RADL0, TEN_RADS0, DSIGMA, Z_AT_W0) */
    DIn u0, v0, t0, qr0, p0, rhoe, dzq, kth0, bbls0, radl0, rads0, zw0;
    DOut w0avg0, tke0;          /* the advanced W0AVG / TKEAVG columns */
    const float* __restrict__ dsigma;   /* DNW(1:nz) */
    DCol rh0;                   /* the driver's RH(i,:,j) */
    DCol q0;                    /* QV1D = MAX(QV,1.E-12), INOUT */
    DOut cldareab, cldliqb, wub;
    /* 1-D outputs (the driver's tendency accumulators) */
    DCol qvten, qcten, qrten, tten, dcaten, qcdcten;
};

#define DENG_SLOT(name, id) DCol name = DWS(wsb, id, kp)

__device__ void deng_shcu(
        const DengColumn& C, float* wsb, int kp, int nz,
        float dt, float dt2, float dx, float dxsq, float xtime, float gmt,
        float zpbl, float ht, float xlong, float capesave, float radsave,
        const float* __restrict__ ainckfsa, int ainc_stride,
        float* rainshv, float* rainsh, float* pblmax, float* cldtopb,
        float* clddpthb, int* ltopb, int* kdcldtop, int* kdcldbas,
        float* capei, float* radiusc, float* ainckfi, int* status) {
    const float PIE = 3.141592654f, TTFRZ = 268.16f, TBFRZ = 248.16f;
    const float RHBC = 0.90f, P00 = 100000.0f, T00 = 273.16f, RLF = 3.339E5f;
    const float AVTS = 11.72f, BVTS = 0.41f, N0R = 8.E6f, XN0 = 1.E-2f;
    const float XMMAX = 9.4E-10f, N0S = 2.E7f, ESI = 0.1f, QCTH = 0.5E-3f;
    const float QCK1 = 1.0E-3f, AVT = 841.99667f, BVT = 0.8f;
    const float RHCRIT = 0.999f;
    const float g_ = DENG_G, r_ = DENG_R, cp_ = DENG_CP;

    DENG_SLOT(qes, 2);    DENG_SLOT(tv0, 3);    DENG_SLOT(dp, 4);
    DENG_SLOT(z0, 5);     DENG_SLOT(dza, 6);    DENG_SLOT(qu, 7);
    DENG_SLOT(tu, 8);     DENG_SLOT(tvu, 9);    DENG_SLOT(umf, 10);
    DENG_SLOT(uer, 11);   DENG_SLOT(udr, 12);   DENG_SLOT(tz, 13);
    DENG_SLOT(qd, 14);    DENG_SLOT(dmf, 15);   DENG_SLOT(der, 16);
    DENG_SLOT(ddr, 17);   DENG_SLOT(tg, 18);    DENG_SLOT(qqg, 19);
    DENG_SLOT(tvg, 20);   DENG_SLOT(ems, 21);   DENG_SLOT(emsd, 22);
    DENG_SLOT(theteu, 23); DENG_SLOT(theted, 24); DENG_SLOT(thetee, 25);
    DENG_SLOT(thtau, 26); DENG_SLOT(thtad, 27); DENG_SLOT(thta0, 28);
    DENG_SLOT(rliq, 29);  DENG_SLOT(rice, 30);  DENG_SLOT(qlqout, 31);
    DENG_SLOT(qicout, 32); DENG_SLOT(pptliq, 33); DENG_SLOT(pptice, 34);
    DENG_SLOT(detlq, 35); DENG_SLOT(detic, 36); DENG_SLOT(umf2, 37);
    DENG_SLOT(dmf2, 38);  DENG_SLOT(detlq2, 39); DENG_SLOT(detic2, 40);
    DENG_SLOT(udr2, 41);  DENG_SLOT(uer2, 42);  DENG_SLOT(der2, 43);
    DENG_SLOT(ddr2, 44);  DENG_SLOT(ratio2, 45); DENG_SLOT(domgdp, 46);
    DENG_SLOT(exn, 47);   DENG_SLOT(tvqu, 48);  DENG_SLOT(eqfrc, 49);
    DENG_SLOT(qdt, 50);   DENG_SLOT(fxm, 51);   DENG_SLOT(thtes, 52);
    DENG_SLOT(thtesg, 53); DENG_SLOT(thtag, 54); DENG_SLOT(thpa, 55);
    DENG_SLOT(qpa, 56);   DENG_SLOT(thfxin, 57); DENG_SLOT(thfxout, 58);
    DENG_SLOT(qfxin, 59); DENG_SLOT(qfxout, 60); DENG_SLOT(dtfm, 61);
    DENG_SLOT(qcg, 62);   DENG_SLOT(qcpa, 63);  DENG_SLOT(qcfxin, 64);
    DENG_SLOT(qcfxout, 65); DENG_SLOT(dca, 66); DENG_SLOT(dcafxin, 67);
    DENG_SLOT(dcafxout, 68); DENG_SLOT(rgvc, 69); DENG_SLOT(faloutc, 70);
    DENG_SLOT(wu, 71);    DENG_SLOT(omg, 72);   DENG_SLOT(kv, 73);
    DENG_SLOT(kr, 74);    DENG_SLOT(lflux, 75); DENG_SLOT(scr3, 76);
    DENG_SLOT(dtdt, 83);  DENG_SLOT(dqdt, 84);  DENG_SLOT(dqldt, 85);
    DENG_SLOT(ddcadt, 86); DENG_SLOT(dqcdcdt, 87);
    const DCol& q0 = C.q0;
    const DCol& rh0 = C.rh0;

    const float gdry = -g_ / cp_;
    const int kl = nz, klm = nz - 1, kx = nz, kxp1 = nz + 1;

    int ltop = 1;
    int kpblmax = 1;
    float clddpth = 0.0f;
    float cldtop = 0.0f;
    for (int k = 1; k <= kl; ++k) {
        wu(k) = 0.0f; omg(k) = 0.0f; umf(k) = 0.0f; thtad(k) = 0.0f;
        kv(k) = 0.0f; kr(k) = 0.0f;
    }
    *rainshv = 0.0f;
    *ainckfi = 0.0f;
    *capei = 0.0f;
    *radiusc = 0.0f;

    const float p300 = C.p0(1) - 30000.0f;
    int ml = 0;
    int l5 = 0;
    for (int k = 1; k <= kx; ++k) {
        float t0k = C.t0(k);
        float es;
        if (t0k > DENG_TO) {
            es = deng_esliq(t0k);
        } else {
            es = deng_esice(t0k);
        }
        qes(k) = (0.622f * es) / (C.p0(k) - es);
        q0(k) = fminf(qes(k), q0(k));
        if (q0(k) > qes(k)) q0(k) = qes(k);
        tv0(k) = t0k * (1.0f + 0.608f * q0(k));
        dp(k) = (C.rhoe(k) * g_) * C.dzq(k);
        if (C.p0(k) >= 500E2f) l5 = k;
        (void)p300;   /* LLFC is never read */
        if (t0k > T00) ml = k;
    }
    z0(1) = 0.5f * C.dzq(1);
    for (int k = 2; k <= kl; ++k) {
        z0(k) = z0(k - 1) + 0.5f * (C.dzq(k) + C.dzq(k - 1));
        dza(k - 1) = z0(k) - z0(k - 1);
    }
    dza(kl) = 0.0f;

    const int lc = 1;
    int kflag = 0;
    float ainc = 999.0f;
    float ctp2 = 99999.9f;
    int ltop2 = 99999;
    int nt = 0;
    float rocpq;
    float wpbltp = 0.0f;
    int klcl = 1;

    /* Everything from the source layer to the end of the closure is one
     * block: each GOTO 425 above label 425 is a `break` out of it, taken
     * only at the block's own level. */
    do {
        int lm1 = 1;
        const float factor1 = 0.20f;
        for (int nk = 1; nk <= kl; ++nk) {
            if ((z0(nk) + 0.5f * C.dzq(nk)) < factor1 * zpbl) {
                lm1 = lm1 + 1;
            } else {
                break;
            }
        }
        int kpbl = 1;
        for (int k = 1; k <= kl; ++k) {
            float zagl_bot = C.zw0(k) - ht;
            float zagl_top = C.zw0(k + 1) - ht;
            if (zpbl >= zagl_bot && zpbl < zagl_top) {
                kpbl = k;
                break;
            }
        }
        int lm2 = 1;
        if (kpbl >= 4) lm2 = 2;
        int lm = max(lm1, lm2);

        float thmix = 0.0f, qmix = 0.0f, zmix = 0.0f, pmix = 0.0f;
        float dpthmx = 0.0f;
        for (int nk = 1; nk <= lm; ++nk) {
            dpthmx = dpthmx + dp(nk);
            rocpq = 0.2854f * (1.0f - 0.28f * q0(nk));
            thmix = thmix + (dp(nk) * C.t0(nk)) * gfk_pow(P00 / C.p0(nk), rocpq);
            qmix = qmix + dp(nk) * q0(nk);
            zmix = zmix + dp(nk) * z0(nk);
            pmix = pmix + dp(nk) * C.p0(nk);
        }
        thmix = thmix / dpthmx;
        qmix = qmix / dpthmx;
        zmix = zmix / dpthmx;
        pmix = pmix / dpthmx;

        rocpq = 0.2854f * (1.0f - 0.28f * qmix);
        float tmix = thmix * gfk_pow(pmix / P00, rocpq);
        float emix = (qmix * pmix) / (0.622f + qmix);
        float tlog = gfk_log((emix / DENG_SVP1) / 1000.0f);
        float tdpt = (DENG_SVP2 * DENG_SVPT0 - DENG_SVP3 * tlog)
                     / (DENG_SVP2 - tlog);
        float tlcl = tdpt - ((0.212f + 1.571E-3f * (tdpt - T00))
                             - 4.36E-4f * (tmix - T00)) * (tmix - tdpt);
        tlcl = fminf(tlcl, tmix);
        float tvlcl = tlcl * (1.0f + 0.608f * qmix);
        float cporq = 1.0f / rocpq;
        float plcl = P00 * gfk_pow(tlcl / thmix, cporq);

        bool found = false;
        for (int nk = 1; nk <= kl; ++nk) {
            klcl = nk;
            if (plcl >= C.p0(nk)) { found = true; break; }
        }
        if (!found) break;                                 /* GOTO 425 */
        int k = klcl - 1;

        if (klcl == 1) {
            /* :822-843.  PPBL/TPBL are only plotted; CLDTOP = ZLCL = Z0(1)
             * reaches CLDTOPB. */
            cldtop = z0(klcl);
            break;                                         /* GOTO 425 */
        }

        float dlp = gfk_log(plcl / C.p0(k)) / gfk_log(C.p0(klcl) / C.p0(k));
        float tenv = C.t0(k) + (C.t0(klcl) - C.t0(k)) * dlp;
        float qenv = q0(k) + (q0(klcl) - q0(k)) * dlp;
        float tven = tenv * (1.0f + 0.608f * qenv);
        float zlcl = z0(k) + (z0(klcl) - z0(k)) * dlp;

        if (zpbl > *pblmax) {
            *pblmax = zpbl;
            kpblmax = kpbl;
        }
        float timeunv = gmt + xtime / 60.0f;
        if (timeunv >= 24.0f) timeunv = timeunv - 24.0f;
        float timeloc = timeunv + xlong / 15.0f;
        if (timeloc < 0.0f) timeloc = timeloc + 24.0f;
        if (timeloc >= 24.0f) timeloc = timeloc - 24.0f;
        if (fabsf(timeloc - 6.0f) < 0.01f) {
            *pblmax = 0.0f;
            kpblmax = 1;
        }
        int kval = min(klcl, kpblmax);
        float emax = 0.0f;
        for (int nk = 1; nk <= kval; ++nk) {
            float val = C.tke0(nk);
            if (val < emax) continue;
            emax = val;
        }
        kval = min(klcl - 1, kpbl);
        float wbar = C.w0avg0(kval);
        float wtke = sqrtf((2.0f * emax) / 3.0f);
        wpbltp = wbar + wtke;
        float wnh = 0.0f;
        if (*clddpthb > 4.0E3f) {
            float zval = zlcl + 2000.0f;
            for (int nk = 1; nk <= kl; ++nk) {
                if (fabsf(z0(nk) - zval) < C.dzq(nk) / 2.0f) kval = nk;
            }
            if (C.wub(kval) > wpbltp) {
                wnh = 0.25f * (C.wub(kval) - wpbltp);
            } else {
                wnh = 0.0f;
            }
        }
        wpbltp = wpbltp + wnh;

        theteu(k) = (tmix * deng_exner_pow(1.E5f / pmix, qmix))
                    * deng_bolton(tlcl, qmix);
        float es = deng_esliq(tenv);
        float qese = (0.622f * es) / (plcl - es);
        thtes(k) = (tenv * deng_exner_pow(1.E5f / plcl, qese))
                   * deng_bolton(tenv, qese);
        float wtw = wpbltp * wpbltp;
        tvlcl = tlcl * (1.0f + 0.608f * qmix);
        float rholcl = plcl / (r_ * tvlcl);
        const int lcl = klcl;
        int let = lcl;

        /* updraft radius, :936-956 */
        float rad;
        {
            float val = *clddpthb;
            float val1 = zpbl;
            if (val / 1000.0f > 3.9f) {
                rad = 1500.0f;
            } else {
                float ff = ((12.0f * val) / (4.0f - val / 1000.0f)) / 1000.0f;
                float h = val1 * ff;
                float b = -(0.5f * (7.0f + (2.0f * h) / 1000.0f));
                rad = 0.5f * (-b - sqrtf(b * b - (12.0f * h) / 1000.0f));
                rad = (rad * 0.5f) * 1000.0f;
                if (rad < 150.0f) rad = 150.0f;
                if (rad > 1500.0f) rad = 1500.0f;
            }
        }
        if (capesave < 100.0f) rad = (rad + radsave) / 2.0f;
        *radiusc = rad;

        /* PBL properties, :960-983 */
        float ppbl, tenvpbl, qenvpbl, tvenpbl;
        if (zpbl <= z0(1)) {
            k = 1;
            ppbl = C.p0(k);
            tenvpbl = C.t0(k);
            qenvpbl = q0(k);
            tvenpbl = tv0(k);
        } else {
            for (int kk = 1; kk <= kl - 1; ++kk) {
                if (zpbl >= z0(kk) && zpbl < z0(kk + 1)) {
                    k = kk;
                    break;
                }
            }
            float c1 = (zpbl - z0(k)) / (z0(k + 1) - z0(k));
            ppbl = C.p0(k) * gfk_exp(c1 * (gfk_log(C.p0(k + 1)) - gfk_log(C.p0(k))));
            tenvpbl = c1 * (C.t0(k + 1) - C.t0(k)) + C.t0(k);
            qenvpbl = c1 * (q0(k + 1) - q0(k)) + q0(k);
            tvenpbl = c1 * (tv0(k + 1) - tv0(k)) + tv0(k);
        }
        (void)qenvpbl;
        int kkpbl = k;
        float tpbl = tlcl * gfk_pow(ppbl / plcl, rocpq);
        float tvpbl = tpbl * (1.0f + 0.608f * qmix);
        float rhopbl = ppbl / (r_ * tvpbl);

        float rho_ini, t_ini, tv_ini, tven_ini, z_ini;
        if (ppbl >= plcl) {
            theteu(k) = (tmix * deng_exner_pow(1.E5f / pmix, qmix))
                        * deng_bolton(tlcl, qmix);
            es = deng_esliq(tenvpbl);
            qese = (0.622f * es) / (ppbl - es);
            thtes(k) = (tenvpbl * deng_exner_pow(1.E5f / ppbl, qese))
                       * deng_bolton(tenvpbl, qese);
            rho_ini = rhopbl;
            t_ini = tpbl;
            tv_ini = tvpbl;
            tven_ini = tvenpbl;
            z_ini = zpbl;
            let = kkpbl;
        } else {
            k = klcl - 1;
            rho_ini = rholcl;
            t_ini = tlcl;
            tv_ini = tvlcl;
            tven_ini = tven;
            z_ini = zlcl;
        }

        /* source-layer depth, :1016-1035 */
        float dpthmx1 = 0.0f;
        for (int nk = 1; nk <= k; ++nk) dpthmx1 = dpthmx1 + dp(nk);
        int llc;
        if (k == 1) {
            llc = 1;
        } else {
            float dpth = ((100.0f * (rad - 150.0f)) / 27.0f) + 1000.0f;
            if (dpth >= dpthmx1) {
                llc = 1;
            } else {
                float pplsdp = (C.p0(k + 1) + dp(k + 1) / 2.0f) + dpth;
                llc = k;
                for (int nk = k - 1; nk >= 1; --nk) {
                    if (fabsf(pplsdp - C.p0(nk)) <= fabsf(pplsdp - C.p0(nk + 1)))
                        llc = nk;
                }
            }
        }

        /* ------------------------------------------------ updraft, :1048 */
        wu(k) = wpbltp;
        const float au0 = (PIE * rad) * rad;
        umf(k) = (rho_ini * au0) * wpbltp;
        const float vmfpblcl = umf(k);
        const float vmflcl = umf(k);
        float upold = vmfpblcl;
        float upnew = upold;
        ratio2(k) = 0.0f;
        uer(k) = 0.0f;
        float abe = 0.0f;
        float trppt = 0.0f;
        tu(k) = t_ini;
        tvu(k) = tv_ini;
        qu(k) = qmix;
        eqfrc(k) = 1.0f;
        rliq(k) = 0.0f; rice(k) = 0.0f; qlqout(k) = 0.0f; qicout(k) = 0.0f;
        detlq(k) = 0.0f; detic(k) = 0.0f; pptliq(k) = 0.0f; pptice(k) = 0.0f;
        int iflag = 0;
        const float thtudl0 = theteu(k);
        float ttemp = TTFRZ;
        float rate = (*clddpthb < 4.E3f) ? 0.0f : 0.01f;
        float ee1 = 1.0f, ud1 = 0.0f, rei = 0.0f;
        float rl = 0.0f;    /* read by TPMIX/ENVIRTHT only once RATIO2 > 0 */
        float effq = 1.0f, qnwfrz = 0.0f;
        float qnewlq = 0.0f, qnewic = 0.0f;
        float udlbe = 0.0f;
        int nk = k;
        for (nk = k; nk <= klm; ++nk) {
            int nk1 = nk + 1;
            ratio2(nk1) = ratio2(nk);
            float frc1 = 0.0f;
            tu(nk1) = C.t0(nk1);
            theteu(nk1) = theteu(nk);
            qu(nk1) = qu(nk);
            rliq(nk1) = rliq(nk);
            rice(nk1) = rice(nk);
            {
                float tuv = tu(nk1), quv = qu(nk1), lq = rliq(nk1), ic = rice(nk1);
                deng_tpmix(C.p0(nk1), theteu(nk1), &tuv, &quv, &lq, &ic,
                           &qnewlq, &qnewic, ratio2(nk1), rl);
                tu(nk1) = tuv; qu(nk1) = quv; rliq(nk1) = lq; rice(nk1) = ic;
            }
            tvu(nk1) = tu(nk1) * (1.0f + 0.608f * qu(nk1));
            if (tu(nk1) <= TTFRZ && iflag < 1) {
                float r1;
                if (tu(nk1) > TBFRZ) {
                    if (ttemp > TTFRZ) ttemp = TTFRZ;
                    frc1 = (ttemp - tu(nk1)) / (TTFRZ - TBFRZ);
                    r1 = (ttemp - tu(nk1)) / (ttemp - TBFRZ);
                } else {
                    frc1 = (ttemp - TBFRZ) / (TTFRZ - TBFRZ);
                    r1 = 1.0f;
                    iflag = 1;
                }
                qnwfrz = qnewlq;
                qnewic = qnewic + (qnewlq * r1) * 0.5f;
                qnewlq = qnewlq - (qnewlq * r1) * 0.5f;
                if (fabsf(ttemp - TBFRZ) < 1.E-3f) {
                    effq = 1.0f;
                } else {
                    effq = (TTFRZ - TBFRZ) / (ttemp - TBFRZ);
                }
                ttemp = tu(nk1);
            }
            float be, boterm, enterm, dzz;
            if (nk == k) {
                be = (tv_ini + tvu(nk1)) / (tven_ini + tv0(nk1)) - 1.0f;
                boterm = (((2.0f * (z0(nk1) - z_ini)) * g_) * be) / 1.5f;
                enterm = 0.0f;
                dzz = z0(nk1) - z_ini;
            } else {
                be = (tvu(nk) + tvu(nk1)) / (tv0(nk) + tv0(nk1)) - 1.0f;
                boterm = (((2.0f * dza(nk)) * g_) * be) / 1.5f;
                enterm = ((2.0f * uer(nk)) * wtw) / upold;
                dzz = dza(nk);
            }
            {
                float lq = rliq(nk1), ic = rice(nk1), lo = qlqout(nk1), io = qicout(nk1);
                deng_condload(&lq, &ic, &wtw, dzz, boterm, enterm, rate,
                              &qnewlq, &qnewic, &lo, &io, g_);
                rliq(nk1) = lq; rice(nk1) = ic; qlqout(nk1) = lo; qicout(nk1) = io;
            }
            if (wtw < 1.E-3f) {
                break;
            } else {
                wu(nk1) = sqrtf(wtw);
            }
            thtes(nk1) = (C.t0(nk1) * deng_exner_pow(1.E5f / C.p0(nk1), qes(nk1)))
                         * deng_bolton(C.t0(nk1), qes(nk1));
            udlbe = ((2.0f * thtudl0) / (thtes(nk) + thtes(nk1)) - 1.0f) * dzz;
            if (udlbe > 0.0f) abe = abe + udlbe * g_;
            if (frc1 > 1.E-6f) {
                float tuv = tu(nk1), th = theteu(nk1), quv = qu(nk1);
                float lq = rliq(nk1), ic = rice(nk1), r2 = ratio2(nk1);
                deng_dtfrznew(&tuv, C.p0(nk1), &th, &quv, &lq, &ic, &r2,
                              &qnwfrz, &rl, &frc1, effq, &iflag);
                tu(nk1) = tuv; theteu(nk1) = th; qu(nk1) = quv;
                rliq(nk1) = lq; rice(nk1) = ic; ratio2(nk1) = r2;
            }
            thetee(nk1) = deng_envirtht(C.p0(nk1), C.t0(nk1), q0(nk1),
                                        ratio2(nk1), rl);
            rei = ((vmfpblcl * dp(nk1)) * 0.03f) / rad;
            tvqu(nk1) = tu(nk1)
                        * (((1.0f + 0.608f * qu(nk1)) - rliq(nk1)) - rice(nk1));
            float ee2, ud2;
            if (tvqu(nk1) <= tv0(nk1)) {
                uer(nk1) = 0.0f;
                udr(nk1) = rei;
                ee2 = 0.0f;
                ud2 = 1.0f;
                eqfrc(nk1) = 0.0f;
            } else {
                let = nk1;
                float ttmp = tvqu(nk1);
                float f1 = 0.95f;
                float f2 = 1.0f - f1;
                float thttmp = f1 * thetee(nk1) + f2 * theteu(nk1);
                float rtmp = f1 * q0(nk1) + f2 * qu(nk1);
                float tmpliq = f2 * rliq(nk1);
                float tmpice = f2 * rice(nk1);
                deng_tpmix(C.p0(nk1), thttmp, &ttmp, &rtmp, &tmpliq, &tmpice,
                           &qnewlq, &qnewic, ratio2(nk1), rl);
                float tu95 = ttmp * (((1.0f + 0.608f * rtmp) - tmpliq) - tmpice);
                if (tu95 > tv0(nk1)) {
                    ee2 = 1.0f;
                    ud2 = 0.0f;
                    eqfrc(nk1) = 1.0f;
                } else {
                    f1 = 0.10f;
                    f2 = 1.0f - f1;
                    thttmp = f1 * thetee(nk1) + f2 * theteu(nk1);
                    rtmp = f1 * q0(nk1) + f2 * qu(nk1);
                    tmpliq = f2 * rliq(nk1);
                    tmpice = f2 * rice(nk1);
                    deng_tpmix(C.p0(nk1), thttmp, &ttmp, &rtmp, &tmpliq,
                               &tmpice, &qnewlq, &qnewic, ratio2(nk1), rl);
                    float tu10 = ttmp * (((1.0f + 0.608f * rtmp) - tmpliq) - tmpice);
                    if (fabsf(tu10 - tvqu(nk1)) < 1.E-3f) {
                        eqfrc(nk1) = 1.0f;
                    } else {
                        eqfrc(nk1) = ((tv0(nk1) - tvqu(nk1)) * f1)
                                     / ((tu10 - tvqu(nk1)) + 1.E-20f);
                    }
                    const float eqmax = 1.0f, eqmin = 0.0f;
                    eqfrc(nk1) = fmaxf(eqmin, eqfrc(nk1));
                    eqfrc(nk1) = fminf(eqmax, eqfrc(nk1));
                    if (eqfrc(nk1) == 1.0f) {
                        ee2 = 1.0f;
                        ud2 = 0.0f;
                    } else if (eqfrc(nk1) == 0.0f) {
                        ee2 = 0.0f;
                        ud2 = 1.0f;
                    } else {
                        deng_prof5(eqfrc(nk1), &ee2, &ud2);
                    }
                }
                /* label 50 */
                uer(nk1) = (0.5f * rei) * (ee1 + ee2);
                udr(nk1) = (0.5f * rei) * (ud1 + ud2);
            }
            /* label 55 */
            if (umf(nk) - udr(nk1) < 10.0f) {
                if (udlbe > 0.0f) abe = abe - udlbe * g_;
                let = nk;
                break;
            }
            ee1 = ee2;
            ud1 = ud2;
            upold = umf(nk) - udr(nk1);
            upnew = upold + uer(nk1);
            umf(nk1) = upnew;
            detlq(nk1) = rliq(nk1) * udr(nk1);
            detic(nk1) = rice(nk1) * udr(nk1);
            qdt(nk1) = qu(nk1);
            qu(nk1) = (upold * qu(nk1) + uer(nk1) * q0(nk1)) / upnew;
            theteu(nk1) = (theteu(nk1) * upold + thetee(nk1) * uer(nk1)) / upnew;
            rliq(nk1) = (rliq(nk1) * upold) / upnew;
            rice(nk1) = (rice(nk1) * upold) / upnew;
            pptliq(nk1) = qlqout(nk1) * umf(nk);
            pptice(nk1) = qicout(nk1) * umf(nk);
            trppt = (trppt + pptliq(nk1)) + pptice(nk1);
        }
        *capei = abe;

        /* updraft top, :1364-1393 */
        int ltop1 = min(nk, klm);
        float ctp1 = z0(ltop1);
        float wumax = wu(1);
        for (int n2 = 2; n2 <= *ltopb; ++n2) {
            if (wu(n2) > wumax) wumax = wu(n2);
        }
        if (*ltopb <= 1) {
            cldtop = ctp1;
            ltop = ltop1;
        } else {
            ctp2 = *cldtopb + (((0.2f * wumax) * dt2) / 2.0f);
            ltop2 = 1;
            if (ctp2 >= (z0(kl) + C.dzq(kl) / 2.0f)) {
                ctp2 = z0(kl);
                ltop2 = kl;
            } else {
                for (int n2 = 1; n2 <= kl; ++n2) {
                    if (ctp2 < (z0(n2) + C.dzq(n2) / 2.0f)
                        && ctp2 >= (z0(n2) - C.dzq(n2) / 2.0f)) ltop2 = n2;
                }
                ctp2 = fmaxf(ctp2, zlcl);
                ltop2 = max(ltop2, klcl);
            }
            cldtop = fminf(ctp1, ctp2);
            ltop = min(ltop1, ltop2);
        }
        if (ltop == ltop2 && ltop1 > ltop2) {
            for (int n2 = ltop2 + 1; n2 <= ltop1; ++n2)
                trppt = (trppt - pptliq(n2)) - pptice(n2);
        }
        clddpth = cldtop - zlcl;
        if (clddpth <= 0.0f) clddpth = 0.0f;

        /* level of free convection, :1418-1478 */
        float zlfc = z0(kl);
        bool to67 = false;
        if (clddpth <= 0.0f) {
            zlfc = z0(kl);
            to67 = true;
        } else if (tvqu(klcl) >= tv0(klcl)) {
            float sum = 0.0f;
            int kv2 = max(klcl, kkpbl);
            for (int ks = kv2; ks <= ltop1; ++ks) {
                if (tvqu(ks) > tv0(ks)) {
                    sum = sum + C.dzq(ks);
                } else {
                    break;
                }
            }
            if (sum >= 300.0f) {
                zlfc = zlcl;
                to67 = true;
            }
        }
        if (!to67) {
            for (int kk = klcl; kk <= ltop1 - 1; ++kk) {
                if (tvqu(kk) < tv0(kk) && tvqu(kk + 1) > tv0(kk + 1)) {
                    int k2 = kk + 1;
                    float val = z0(k2);
                    float sum = 0.0f;
                    for (int ks = k2; ks <= ltop1; ++ks) {
                        if (tvqu(ks) > tv0(ks)) {
                            sum = sum + C.dzq(ks);
                        } else {
                            break;
                        }
                    }
                    if (sum >= 300.0f) {
                        zlfc = val;
                        break;
                    }
                }
            }
        }
        zlfc = fmaxf(zlfc, zpbl);

        if (cldtop <= zlcl || ltop < klcl) {
            cldtop = z0(1);
            ltop = 1;
            break;                                         /* GOTO 425 */
        }
        if (let > ltop) let = ltop;
        if (let == ltop) {
            upold = umf(ltop - 1) - udr(ltop);
            upnew = upold + uer(ltop);
            udr(ltop) = (umf(ltop) + udr(ltop)) - uer(ltop);
            if (fabsf(upold) < 0.001f) {
                *status |= DENG_STATUS_STOP907;
                return;
            }
            detlq(ltop) = ((rliq(ltop) * udr(ltop)) * upnew) / upold;
            detic(ltop) = ((rice(ltop) * udr(ltop)) * upnew) / upold;
            uer(ltop) = 0.0f;
            umf(ltop) = 0.0f;
        } else {
            float dptt = 0.0f;
            for (int nj = let + 1; nj <= ltop; ++nj) dptt = dptt + dp(nj);
            float dumfdp = umf(let) / dptt;
            for (int n2 = let + 1; n2 <= ltop; ++n2) {
                udr(n2) = dp(n2) * dumfdp;
                umf(n2) = umf(n2 - 1) - udr(n2);
                detlq(n2) = rliq(n2) * udr(n2);
                detic(n2) = rice(n2) * udr(n2);
                if (n2 >= let + 2) {
                    trppt = (trppt - pptliq(n2)) - pptice(n2);
                    pptliq(n2) = umf(n2 - 1) * qlqout(n2);
                    pptice(n2) = umf(n2 - 1) * qicout(n2);
                    trppt = (trppt + pptliq(n2)) + pptice(n2);
                }
            }
        }
        /* label 85: below the source layer, :1552-1604 */
        float qmix1 = 0.0f, zmix1 = 0.0f;
        dpthmx1 = 0.0f;
        for (int n2 = llc; n2 <= k; ++n2) {
            dpthmx1 = dpthmx1 + dp(n2);
            qmix1 = qmix1 + dp(n2) * q0(n2);
            zmix1 = zmix1 + dp(n2) * z0(n2);
        }
        qmix1 = qmix1 / dpthmx1;
        zmix1 = zmix1 / dpthmx1;
        for (int n2 = 1; n2 <= k; ++n2) {
            if (n2 >= llc) {
                if (n2 == llc) {
                    umf(n2) = (vmfpblcl * dp(n2)) / dpthmx1;
                    uer(n2) = (vmfpblcl * dp(n2)) / dpthmx1;
                } else {
                    /* NK.LE.K always holds here; the ELSE arm (:1571-1573)
                     * is unreachable. */
                    uer(n2) = (vmfpblcl * dp(n2)) / dpthmx1;
                    umf(n2) = umf(n2 - 1) + uer(n2);
                }
                qu(n2) = qmix1;
                tu(n2) = tlcl + (z0(n2) - zlcl) * gdry;
                wu(n2) = wpbltp;
            } else {
                tu(n2) = 0.0f; qu(n2) = 0.0f; umf(n2) = 0.0f; wu(n2) = 0.0f;
                uer(n2) = 0.0f;
            }
            udr(n2) = 0.0f; qdt(n2) = 0.0f; rliq(n2) = 0.0f; rice(n2) = 0.0f;
            qlqout(n2) = 0.0f; qicout(n2) = 0.0f; pptliq(n2) = 0.0f;
            pptice(n2) = 0.0f; detlq(n2) = 0.0f; detic(n2) = 0.0f;
            ratio2(n2) = 0.0f;
            float ee = (q0(n2) * C.p0(n2)) / (0.622f + q0(n2));
            float tl = gfk_log((ee / DENG_SVP1) / 1000.0f);
            float td = (DENG_SVP2 * DENG_SVPT0 - DENG_SVP3 * tl) / (DENG_SVP2 - tl);
            float tsat = td - ((0.212f + 1.571E-3f * (td - T00))
                               - 4.36E-4f * (C.t0(n2) - T00)) * (C.t0(n2) - td);
            float thta = C.t0(n2) * deng_exner_pow(1.E5f / C.p0(n2), q0(n2));
            thetee(n2) = thta * deng_bolton(tsat, q0(n2));
            thtes(n2) = thta * deng_bolton(C.t0(n2), qes(n2));
            eqfrc(n2) = 1.0f;
        }
        ltop1 = ltop + 1;
        const int ltopm1 = ltop - 1;
        for (int n2 = ltop1; n2 <= kx; ++n2) {
            umf(n2) = 0.0f; udr(n2) = 0.0f; uer(n2) = 0.0f; qdt(n2) = 0.0f;
            rliq(n2) = 0.0f; rice(n2) = 0.0f; qlqout(n2) = 0.0f;
            qicout(n2) = 0.0f; detlq(n2) = 0.0f; detic(n2) = 0.0f;
            pptliq(n2) = 0.0f; pptice(n2) = 0.0f;
            if (n2 > ltop1) {
                tu(n2) = 0.0f; qu(n2) = 0.0f; wu(n2) = 0.0f;
            }
            thta0(n2) = 0.0f; thtau(n2) = 0.0f;
            ems(n2) = (dp(n2) * dxsq) / g_;
            emsd(n2) = 1.0f / ems(n2);
            tg(n2) = C.t0(n2); qqg(n2) = q0(n2); qcg(n2) = 0.0f; dca(n2) = 0.0f;
            dtfm(n2) = 0.0f; fxm(n2) = 0.0f; omg(n2) = 0.0f;
        }
        omg(kxp1) = 0.0f;

        const float p165 = C.p0(klcl) - 1.65E4f;
        float pptmlt = 0.0f;
        int lvf = 0;
        for (int n2 = 1; n2 <= ltop; ++n2) {
            ems(n2) = (dp(n2) * dxsq) / g_;
            emsd(n2) = 1.0f / ems(n2);
            dtfm(n2) = 0.0f;
            exn(n2) = deng_exner_pow(P00 / C.p0(n2), qdt(n2));
            thtau(n2) = tu(n2) * exn(n2);
            exn(n2) = deng_exner_pow(P00 / C.p0(n2), q0(n2));
            thta0(n2) = C.t0(n2) * exn(n2);
            if (C.p0(n2) > p165) lvf = n2;
            pptmlt = pptmlt + pptice(n2);
            omg(n2) = 0.0f;
        }
        float clvf = 0.0f;
        for (int n2 = klcl; n2 <= lvf + 1; ++n2)
            clvf = (clvf + pptliq(n2)) + pptice(n2);
        float usr = umf(lvf + 1) * qu(lvf + 1) + clvf;
        usr = fminf(usr, trppt);

        /* convective time scale, :1694-1705.  L5 is the highest level at or
         * below 500 hPa; a column whose lowest level is above it has no L5
         * and WRF reads U0(0). */
        if (l5 < 1) {
            *status |= DENG_STATUS_NO_500HPA;
            return;
        }
        float wspd_klcl = sqrtf(C.u0(klcl) * C.u0(klcl) + C.v0(klcl) * C.v0(klcl));
        float wspd_l5 = sqrtf(C.u0(l5) * C.u0(l5) + C.v0(l5) * C.v0(l5));
        float wspd_ltop = sqrtf(C.u0(ltop) * C.u0(ltop) + C.v0(ltop) * C.v0(ltop));
        float vconv = 0.5f * (wspd_klcl + wspd_l5);
        float timec = dx / vconv;
        const float tcmin = 1800.0f, tcmax = 3600.0f;
        timec = fmaxf(tcmin, timec);
        timec = fminf(tcmax, timec);
        int nic = deng_nint(timec / (0.5f * dt2));
        timec = ((float)nic * 0.5f) * dt2;

        float shsign = (wspd_ltop > wspd_klcl) ? 1.0f : -1.0f;
        float vws = (C.u0(ltop) - C.u0(klcl)) * (C.u0(ltop) - C.u0(klcl))
                    + (C.v0(ltop) - C.v0(klcl)) * (C.v0(ltop) - C.v0(klcl));
        if (ltop <= lcl) break;                            /* GOTO 425 */
        vws = ((1.E3f * shsign) * sqrtf(vws)) / (z0(ltop) - z0(lcl));
        float pef = 1.591f + vws * (-0.639f + vws * (9.53E-2f - vws * 4.96E-3f));
        const float pefmx = 0.9f, pefmn = 0.2f;
        pef = fmaxf(pef, pefmn);
        pef = fminf(pef, pefmx);
        float cbh = (zlcl - z0(1)) * 3.281E-3f;
        float rcbh;
        if (cbh < 3.0f) {
            rcbh = 0.02f;
        } else {
            rcbh = 0.96729352f + cbh * (-0.70034167f + cbh * (0.162179896f
                   + cbh * (-1.2569798E-2f + cbh * (4.2772E-4f - cbh * 5.44E-6f))));
        }
        if (cbh > 25.0f) rcbh = 2.4f;
        float pefcbh = 1.0f / (1.0f + rcbh);
        const float cbhmx = 0.9f;
        pefcbh = fminf(pefcbh, cbhmx);
        float peff = 0.5f * (pef + pefcbh);
        if (*clddpthb < 4.E3f) {
            usr = trppt;
            peff = 0.99999f;
        }

        /* ---------------------------------------------- downdraft, :1764 */
        float tder = 0.0f;
        int kstart = max(kkpbl, klcl);
        float thtmin = thtes(kstart + 1);
        int kmin = kstart + 1;
        for (int n2 = kstart + 2; n2 <= ltop - 1; ++n2) {
            thtmin = fminf(thtmin, thtes(n2));
            if (thtmin == thtes(n2)) kmin = n2;
        }
        int lfs = kmin;
        if (lfs >= ltop) lfs = ltop - 1;
        if (ratio2(lfs) > 0.0f)
            thetee(lfs) = deng_envirtht(C.p0(lfs), C.t0(lfs), q0(lfs), 0.0f, rl);
        if (fabsf(thetee(lfs) - theteu(lfs)) < 1.E-3f) {
            eqfrc(lfs) = 1.0f;
        } else {
            eqfrc(lfs) = (thtes(lfs) - theteu(lfs)) / (thetee(lfs) - theteu(lfs));
        }
        eqfrc(lfs) = fmaxf(eqfrc(lfs), 0.0f);
        eqfrc(lfs) = fminf(eqfrc(lfs), 1.0f);
        theted(lfs) = thtes(lfs);
        float dtmltd;
        if (ml > 0) {
            dtmltd = ((0.5f * (qu(klcl) - qu(ltop))) * RLF) / cp_;
        } else {
            dtmltd = 0.0f;
        }
        tz(lfs) = C.t0(lfs) - dtmltd;
        es = deng_esliq(tz(lfs));
        float qs = (0.622f * es) / (C.p0(lfs) - es);
        qd(lfs) = eqfrc(lfs) * q0(lfs) + (1.0f - eqfrc(lfs)) * qu(lfs);
        thtad(lfs) = tz(lfs) * deng_exner_pow(P00 / C.p0(lfs), qd(lfs));
        if (qd(lfs) >= qs) {
            theted(lfs) = thtad(lfs) * deng_bolton(tz(lfs), qs);
        } else {
            theted(lfs) = deng_envirtht(C.p0(lfs), tz(lfs), qd(lfs), 0.0f, rl);
        }
        int ldb = 1;
        for (int n2 = 1; n2 <= lfs; ++n2) {
            int nd = lfs - n2;
            if (theted(lfs) > thtes(nd) || nd == 1) {
                ldb = nd;
                break;
            }
        }
        float dpdd = 20.0E2f;
        float dpt = 0.0f;
        int ldt = ldb;
        float frc = 1.0f;
        for (int n2 = ldb; n2 <= lfs; ++n2) {
            dpt = dpt + dp(n2);
            if (dpt > dpdd) {
                ldt = n2;
                frc = ((dpdd + dp(n2)) - dpt) / dp(n2);
                break;
            }
            if (n2 == lfs - 1) {
                ldt = n2;
                frc = 1.0f;
                dpdd = dpt;
                break;
            }
        }
        if (ldb == lfs) {
            tder = 0.0f;
        } else {
            float tvd_lfs = C.t0(lfs) * (1.0f + 0.608f * qes(lfs));
            float rdd = C.p0(lfs) / (r_ * tvd_lfs);
            float a1 = (1.0f - peff) * au0;
            dmf(lfs) = -(a1 * rdd);
            der(lfs) = eqfrc(lfs) * dmf(lfs);
            ddr(lfs) = 0.0f;
            for (int nd = lfs - 1; nd >= ldb; --nd) {
                int nd1 = nd + 1;
                if (nd <= ldt) {
                    der(nd) = 0.0f;
                    ddr(nd) = -(((dmf(ldt + 1) * dp(nd)) * frc) / dpdd);
                    dmf(nd) = dmf(nd1) + ddr(nd);
                    frc = 1.0f;
                    theted(nd) = theted(nd1);
                    qd(nd) = qd(nd1);
                } else {
                    der(nd) = ((dmf(lfs) * 0.03f) * dp(nd)) / rad;
                    ddr(nd) = 0.0f;
                    dmf(nd) = dmf(nd1) + der(nd);
                    if (ratio2(nd) > 0.0f)
                        thetee(nd) = deng_envirtht(C.p0(nd), C.t0(nd), q0(nd), 0.0f, rl);
                    theted(nd) = (theted(nd1) * dmf(nd1) + thetee(nd) * der(nd)) / dmf(nd);
                    qd(nd) = (qd(nd1) * dmf(nd1) + q0(nd) * der(nd)) / dmf(nd);
                }
            }
            tder = 0.0f;
            for (int nd = ldb; nd <= ldt; ++nd) {
                tz(nd) = deng_tpddbg(C.p0(nd), theted(ldt), C.t0(nd), &qs);
                es = deng_esliq(tz(nd));
                qs = (0.622f * es) / (C.p0(nd) - es);
                float dqssdt = (DENG_SVP2 * (DENG_SVPT0 - DENG_SVP3))
                               / ((tz(nd) - DENG_SVP3) * (tz(nd) - DENG_SVP3));
                rl = DENG_XLV0 - DENG_XLV1 * tz(nd);
                float dtmp = ((rl * qs) * (1.0f - RHBC))
                             / (cp_ + ((rl * RHBC) * qs) * dqssdt);
                float t1rh = tz(nd) + dtmp;
                es = ((RHBC * 1.E3f) * DENG_SVP1)
                     * gfk_exp((DENG_SVP2 * (t1rh - DENG_SVPT0)) / (t1rh - DENG_SVP3));
                float qsrh = (0.622f * es) / (C.p0(nd) - es);
                if (qsrh < qd(nd)) {
                    qsrh = qd(nd);
                    t1rh = tz(nd);
                }
                tz(nd) = t1rh;
                qs = qsrh;
                tder = tder + (qs - qd(nd)) * ddr(nd);
                qd(nd) = qs;
                thtad(nd) = tz(nd) * deng_exner_pow(P00 / C.p0(nd), qd(nd));
            }
        }
        /* label 141 */
        float pptflx = 0.0f, cpr = 0.0f, updinc = 1.0f, ppr = 0.0f;
        for (;;) {
            if (tder < 1.0f) {
                pptflx = trppt;
                cpr = trppt;
                tder = 0.0f;
                updinc = 1.0f;
                ldb = lfs;
                for (int ndk = 1; ndk <= kx; ++ndk) {
                    dmf(ndk) = 0.0f; der(ndk) = 0.0f; ddr(ndk) = 0.0f;
                    thtad(ndk) = 0.0f; tz(ndk) = 0.0f; qd(ndk) = 0.0f;
                }
                break;                                     /* GOTO 165 */
            }
            float devdmf = tder / dmf(lfs);
            ppr = 0.0f;
            pptflx = peff * usr;
            float rced = trppt - pptflx;
            for (int nm = klcl; nm <= lfs; ++nm)
                ppr = (ppr + pptliq(nm)) + pptice(nm);
            float dpptdf;
            if (lfs >= klcl) {
                dpptdf = (((1.0f - peff) * ppr) * (1.0f - eqfrc(lfs))) / umf(lfs);
            } else {
                dpptdf = 0.0f;
            }
            float cndtnf = (rliq(lfs) + rice(lfs)) * (1.0f - eqfrc(lfs));
            float dmflfs = rced / ((devdmf + dpptdf) + cndtnf);
            if (dmflfs > 0.0f) {
                tder = 0.0f;
                continue;                                  /* GOTO 141 */
            }
            float ddinc = dmflfs / dmf(lfs);
            if (lfs >= klcl) {
                updinc = (umf(lfs) - (1.0f - eqfrc(lfs)) * dmflfs) / umf(lfs);
            } else {
                updinc = 1.0f;
            }
            for (int n2 = ldb; n2 <= lfs; ++n2) {
                dmf(n2) = dmf(n2) * ddinc;
                der(n2) = der(n2) * ddinc;
                ddr(n2) = ddr(n2) * ddinc;
            }
            cpr = trppt + ppr * (updinc - 1.0f);
            pptflx = pptflx + (peff * ppr) * (updinc - 1.0f);
            tder = tder * ddinc;
            for (int n2 = lc; n2 <= lfs; ++n2) {
                umf(n2) = umf(n2) * updinc;
                udr(n2) = udr(n2) * updinc;
                uer(n2) = uer(n2) * updinc;
                pptliq(n2) = pptliq(n2) * updinc;
                pptice(n2) = pptice(n2) * updinc;
                detlq(n2) = detlq(n2) * updinc;
                detic(n2) = detic(n2) * updinc;
            }
            if (ldb > 1) {
                for (int n2 = 1; n2 <= ldb - 1; ++n2) {
                    dmf(n2) = 0.0f; der(n2) = 0.0f; ddr(n2) = 0.0f;
                    tz(n2) = 0.0f; qd(n2) = 0.0f; thtad(n2) = 0.0f;
                }
            }
            for (int n2 = lfs + 1; n2 <= kx; ++n2) {
                dmf(n2) = 0.0f; der(n2) = 0.0f; ddr(n2) = 0.0f;
                tz(n2) = 0.0f; qd(n2) = 0.0f; thtad(n2) = 0.0f;
            }
            for (int n2 = ldt + 1; n2 <= lfs - 1; ++n2) {
                tz(n2) = 0.0f;
                qd(n2) = 0.0f;
            }
            break;
        }

        /* label 165: inflow limits, :2035-2044.  AINCM1 is set on the first
         * pass (UER(LLC) > 0 >= DER(LLC) for any updraft that got here). */
        float aincmx = 1000.0f;
        const int lmax = max(klcl, lfs);
        float aincm1 = __int_as_float(0x7fc00000);
        for (int n2 = llc; n2 <= lmax; ++n2) {
            if ((uer(n2) - der(n2)) > 0.0f)
                aincm1 = ems(n2) / ((uer(n2) - der(n2)) * timec);
            aincmx = fminf(aincmx, aincm1);
        }
        ainc = 1.0f;
        if (aincmx < ainc) ainc = aincmx;

        int ncount = 0;
        pptmlt = 0.0f;
        float tder2 = tder;
        float pptfl2 = pptflx;
        for (int n2 = 1; n2 <= ltop; ++n2) {
            detlq2(n2) = detlq(n2); detic2(n2) = detic(n2);
            udr2(n2) = udr(n2); uer2(n2) = uer(n2); ddr2(n2) = ddr(n2);
            der2(n2) = der(n2); umf2(n2) = umf(n2); dmf2(n2) = dmf(n2);
            if (n2 > ml && n2 < ltop) pptmlt = pptmlt + pptice(n2 + 1);
        }
        float pptml2 = pptmlt;
        float fabe = 1.0f;
        const float stab = 0.95f;
        float ainc2 = ainc;
        float abeg = 0.0f;
        bool enter255 = (ainc / aincmx > 0.999f);
        if (!enter255) {
            kkpbl = k;
            float tkemax = 0.0f;
            for (int kk = llc; kk <= klcl; ++kk) tkemax = fmaxf(tkemax, C.tke0(kk));
            tkemax = fminf(tkemax, 10.0f);
            tkemax = fmaxf(tkemax, 1.0f);
            float dsource = (z0(k) - z0(llc)) + 0.5f * (C.dzq(k) + C.dzq(llc));
            float timecs = (dsource / wpbltp) * 45.0f;
            ainc2 = ((tkemax * dpthmx1) * dxsq) / ((vmflcl * g_) * timecs);
            if (*capei < 100.0f) {
                nt = deng_nint((15.0f * 60.0f) / (0.5f * dt)) - 1;
                nt = min(nt, 100);
            } else {
                nt = 0;
            }
        }

        /* ------------------------------------------ closure loop, :2105 */
        for (;;) {
            if (!enter255) {
                int next = 0;   /* 255 or 265 */
                do {
                    ncount = ncount + 1;
                    float dtt = timec;
                    for (int n2 = 1; n2 <= ltop; ++n2) {
                        domgdp(n2) = -((((uer(n2) - der(n2)) - udr(n2)) - ddr(n2)) * emsd(n2));
                        if (n2 > 1) {
                            omg(n2) = omg(n2 - 1) - dp(n2 - 1) * domgdp(n2 - 1);
                            float dtt1 = (0.75f * dp(n2 - 1)) / (fabsf(omg(n2)) + 1.E-10f);
                            dtt = fminf(dtt, dtt1);
                        }
                    }
                    if (kflag != 2) {
                        if ((clddpth + zlcl) <= zlfc) {
                            ainc = ainc2;
                            kflag = 2;
                            next = 255;
                            break;
                        }
                    }
                    /* label 176 */
                    int nstep = 1;
                    float dtime = timec;
                    for (int n2 = 1; n2 <= ltop; ++n2) {
                        thpa(n2) = thta0(n2);
                        qpa(n2) = q0(n2);
                        qcpa(n2) = 0.0f;          /* QC0 */
                        dca(n2) = 0.0f;           /* DCA0 */
                        nstep = deng_nint(timec / dtt + 1.0f);
                        dtime = timec / (float)nstep;
                        fxm(n2) = (omg(n2) * dxsq) / g_;
                    }
                    for (int ntc = 1; ntc <= nstep; ++ntc) {
                        for (int n2 = 1; n2 <= ltop; ++n2) {
                            thfxin(n2) = 0.0f; thfxout(n2) = 0.0f;
                            qfxin(n2) = 0.0f; qfxout(n2) = 0.0f;
                            qcfxin(n2) = 0.0f; qcfxout(n2) = 0.0f;
                            dcafxin(n2) = 0.0f; dcafxout(n2) = 0.0f;
                        }
                        for (int n2 = 2; n2 <= ltop; ++n2) {
                            if (omg(n2) <= 0.0f) {
                                thfxin(n2) = -(fxm(n2) * thpa(n2 - 1));
                                qfxin(n2) = -(fxm(n2) * qpa(n2 - 1));
                                qcfxin(n2) = -(fxm(n2) * qcpa(n2 - 1));
                                dcafxin(n2) = -(fxm(n2) * dca(n2 - 1));
                                thfxout(n2 - 1) = thfxout(n2 - 1) + thfxin(n2);
                                qfxout(n2 - 1) = qfxout(n2 - 1) + qfxin(n2);
                                qcfxout(n2 - 1) = qcfxout(n2 - 1) + qcfxin(n2);
                                dcafxout(n2 - 1) = dcafxout(n2 - 1) + dcafxin(n2);
                            } else {
                                float gd = g_ / cp_;
                                float cpm = cp_ * (1.0f + 0.887f * q0(n2));
                                float rlx, dqssdt;
                                if (C.t0(n2) > DENG_TO) {
                                    rlx = DENG_XLV0 - DENG_XLV1 * C.t0(n2);
                                    dqssdt = ((qes(n2) * DENG_SVP2) * (DENG_SVPT0 - DENG_SVP3))
                                             / ((C.t0(n2) - DENG_SVP3) * (C.t0(n2) - DENG_SVP3));
                                } else {
                                    rlx = DENG_XLS;
                                    dqssdt = (qes(n2) * 6.15E3f) / (C.t0(n2) * C.t0(n2));
                                }
                                rl = rlx;
                                float gm = gd / (1.0f + (rl / cpm) * dqssdt);
                                float dtempdt = (((gd - gm) * omg(n2)) / C.rhoe(n2)) / g_;
                                if ((qcpa(n2) - ((dtempdt * cp_) / rl) * dtime) <= 0.0f)
                                    dtempdt = qcpa(n2) / ((cp_ / rl) * dtime);
                                float dthetadt = dtempdt * deng_exner_pow(P00 / C.p0(n2), q0(n2));
                                thfxout(n2) = fxm(n2) * (thpa(n2) - dthetadt * dtime);
                                qfxout(n2) = fxm(n2) * (qpa(n2) + ((dtempdt * cp_) / rl) * dtime);
                                qcfxout(n2) = fxm(n2) * (qcpa(n2) - ((dtempdt * cp_) / rl) * dtime);
                                dcafxout(n2) = fxm(n2) * dca(n2);
                                thfxin(n2 - 1) = thfxin(n2 - 1) + thfxout(n2);
                                qfxin(n2 - 1) = qfxin(n2 - 1) + qfxout(n2);
                                qcfxin(n2 - 1) = qcfxin(n2 - 1) + qcfxout(n2);
                                dcafxin(n2 - 1) = dcafxin(n2 - 1) + dcafxout(n2);
                            }
                        }
                        for (int n2 = ltop; n2 >= 1; --n2) {
                            if (n2 > kkpbl || n2 < llc) {
                                thpa(n2) = thpa(n2)
                                    + ((((((thfxin(n2) + udr(n2) * thtau(n2)) + ddr(n2) * thtad(n2))
                                          - thfxout(n2)) - (uer(n2) - der(n2)) * thta0(n2))
                                        * dtime) * emsd(n2));
                                qpa(n2) = qpa(n2)
                                    + ((((((qfxin(n2) + udr(n2) * qdt(n2)) + ddr(n2) * qd(n2))
                                          - qfxout(n2)) - (uer(n2) - der(n2)) * q0(n2))
                                        * dtime) * emsd(n2));
                            } else {
                                thpa(n2) = thpa(n2)
                                    + (((((((thfxin(n2) + udr(n2) * thtau(n2)) + ddr(n2) * thtad(n2))
                                           - thfxout(n2)) - uer(n2) * thmix) + der(n2) * thta0(n2))
                                        * dtime) * emsd(n2));
                                qpa(n2) = qpa(n2)
                                    + (((((((qfxin(n2) + udr(n2) * qdt(n2)) + ddr(n2) * qd(n2))
                                           - qfxout(n2)) - uer(n2) * qmix) + der(n2) * q0(n2))
                                        * dtime) * emsd(n2));
                            }
                            qcpa(n2) = qcpa(n2)
                                + (((((qcfxin(n2) + detlq(n2)) + detic(n2)) - qcfxout(n2))
                                    * dtime) * emsd(n2));
                            dca(n2) = dca(n2)
                                + ((((dcafxin(n2) - dcafxout(n2)) + udr(n2)) * dtime) * emsd(n2));
                        }
                    }
                    for (int n2 = ltop; n2 >= 1; --n2) {
                        thtag(n2) = thpa(n2);
                        qqg(n2) = qpa(n2);
                        qcg(n2) = qcpa(n2);
                    }
                    for (int n2 = 2; n2 <= ltop; ++n2) {
                        if (qqg(n2) < 0.0f) {
                            float tma = qqg(n2 + 1) * ems(n2 + 1);
                            float tmb = qqg(n2 - 1) * ems(n2 - 1);
                            float tmm = (qqg(n2) - 1.E-9f) * ems(n2);
                            float bcoeff = 0.0f, acoeff = 0.0f;
                            if (tma != 0.0f && tmb != 0.0f) {
                                bcoeff = -(tmm / ((tma * tma) / tmb + tmb));
                                acoeff = (bcoeff * tma) / tmb;
                            } else if (tma == 0.0f && tmb == 0.0f) {
                                bcoeff = 0.0f;
                                acoeff = 0.0f;
                            } else if (tma == 0.0f) {
                                bcoeff = -(tmm / tmb);
                                acoeff = 0.0f;
                            } else if (tmb == 0.0f) {
                                bcoeff = 0.0f;
                                acoeff = -(tmm / tma);
                            }
                            tmb = tmb * (1.0f - bcoeff);
                            tma = tma * (1.0f - acoeff);
                            qqg(n2) = 1.E-9f;
                            qqg(n2 + 1) = tma * emsd(n2 + 1);
                            qqg(n2 - 1) = tmb * emsd(n2 - 1);
                        }
                    }
                    float topomg = ((udr(ltop) - uer(ltop)) * dp(ltop)) * emsd(ltop);
                    if (fabsf(topomg - omg(ltop)) > 1.E-3f) {
                        /* ISTOP = 1: WRF leaves TG(1:LTOP) as the previous
                         * pass left it, unset on the first. */
                        *status |= DENG_STATUS_MASS_IMBALANCE;
                        next = 265;
                        break;
                    }
                    for (int n2 = 1; n2 <= ltop; ++n2) {
                        exn(n2) = deng_exner_pow(P00 / C.p0(n2), qqg(n2));
                        tg(n2) = thtag(n2) / exn(n2);
                        if (n2 > ml) dtfm(n2) = ((detlq(n2) * RLF) * emsd(n2)) / cp_;
                        tg(n2) = tg(n2) + dtfm(n2) * timec;
                        tvg(n2) = tg(n2) * (1.0f + 0.608f * qqg(n2));
                    }
                    {
                        const float dtmlte = 0.0f;
                        float tdp = 0.0f;
                        for (int kk = 1; kk <= ml; ++kk) {
                            int n2 = ml - kk + 1;
                            tdp = tdp + dp(n2);
                            if (tdp < 2.E4f) {
                                dtfm(n2) = dtmlte;
                                tg(n2) = tg(n2) + dtmlte * timec;
                            } else {
                                dtfm(n2) = (dtmlte * ((2.E4f + dp(n2)) - tdp)) / dp(n2);
                                tg(n2) = tg(n2) + ((dtmlte * timec) * ((2.E4f + dp(n2)) - tdp)) / dp(n2);
                                break;
                            }
                        }
                    }
                    if (kflag == 2 || kflag == 3) {
                        next = 265;
                        break;
                    }
                    /* new cloud and the change in ABE, :2325-2412 */
                    float thmix2 = 0.0f, qmix2 = 0.0f, pmix2 = 0.0f;
                    for (int n2 = 1; n2 <= lm; ++n2) {
                        rocpq = 0.2854f * (1.0f - 0.28f * qqg(n2));
                        thmix2 = thmix2 + (dp(n2) * tg(n2)) * gfk_pow(P00 / C.p0(n2), rocpq);
                        qmix2 = qmix2 + dp(n2) * qqg(n2);
                        pmix2 = pmix2 + dp(n2) * C.p0(n2);
                    }
                    thmix2 = thmix2 / dpthmx;
                    qmix2 = qmix2 / dpthmx;
                    pmix2 = pmix2 / dpthmx;
                    rocpq = 0.2854f * (1.0f - 0.28f * qmix2);
                    float tmix2 = thmix2 * gfk_pow(pmix2 / P00, rocpq);
                    es = deng_esliq(tmix2);
                    qs = (0.622f * es) / (pmix2 - es);
                    float ttlcl, pplcl;
                    if (qmix2 > qs) {
                        rl = DENG_XLV0 - DENG_XLV1 * tmix2;
                        float cpm = cp_ * (1.0f + 0.887f * qmix2);
                        float dqssdt = ((qs * DENG_SVP2) * (DENG_SVPT0 - DENG_SVP3))
                                       / ((tmix2 - DENG_SVP3) * (tmix2 - DENG_SVP3));
                        float dq = (qmix2 - qs) / (1.0f + (rl * dqssdt) / cpm);
                        tmix2 = tmix2 + (rl / cp_) * dq;
                        qmix2 = qmix2 - dq;
                        rocpq = 0.2854f * (1.0f - 0.28f * qmix2);
                        thmix2 = tmix2 * gfk_pow(P00 / pmix2, rocpq);
                        ttlcl = tmix2;
                        pplcl = pmix2;
                    } else {
                        const float qmin = 0.0f;
                        qmix2 = fmaxf(qmix2, qmin);
                        float emix2 = (qmix2 * pmix2) / (0.622f + qmix2);
                        float tl = gfk_log((emix2 / DENG_SVP1) / 1000.0f);
                        float td = (DENG_SVP2 * DENG_SVPT0 - DENG_SVP3 * tl) / (DENG_SVP2 - tl);
                        ttlcl = td - ((0.212f + 1.571E-3f * (td - T00))
                                      - 4.36E-4f * (tmix2 - T00)) * (tmix2 - td);
                        ttlcl = fminf(ttlcl, tmix2);
                        cporq = 1.0f / rocpq;
                        pplcl = P00 * gfk_pow(ttlcl / thmix2, cporq);
                    }
                    for (int n2 = lc; n2 <= kl; ++n2) {
                        klcl = n2;
                        if (pplcl >= C.p0(n2)) break;
                    }
                    k = klcl - 1;
                    if (k < 1) {
                        /* WRF reads P0(0), TG(0), ... past the array. */
                        *status |= DENG_STATUS_LCL_AT_SURFACE;
                        return;
                    }
                    dlp = gfk_log(pplcl / C.p0(k)) / gfk_log(C.p0(klcl) / C.p0(k));
                    tenv = tg(k) + (tg(klcl) - tg(k)) * dlp;
                    qenv = qqg(k) + (qqg(klcl) - qqg(k)) * dlp;
                    tven = tenv * (1.0f + 0.608f * qenv);
                    float tvbar = 0.5f * (tvg(k) + tven);
                    float zlcl1 = z0(k) + ((r_ * tvbar) * gfk_log(C.p0(k) / pplcl)) / g_;
                    float tvavg = 0.5f * (tven + tg(klcl) * (1.0f + 0.608f * qqg(klcl)));
                    pplcl = C.p0(klcl) * gfk_exp((g_ / (r_ * tvavg)) * (z0(klcl) - zlcl1));
                    theteu(k) = (tmix2 * deng_exner_pow(1.E5f / pmix2, qmix2))
                                * deng_bolton(ttlcl, qmix2);
                    es = deng_esliq(tenv);
                    qese = (0.622f * es) / (pplcl - es);
                    thtesg(k) = (tenv * deng_exner_pow(1.E5f / pplcl, qese))
                                * deng_bolton(tenv, qese);
                    abeg = 0.0f;
                    float thtudl = theteu(k);
                    for (int n2 = k; n2 <= ltopm1; ++n2) {
                        int n21 = n2 + 1;
                        es = deng_esliq(tg(n21));
                        qese = (0.622f * es) / (C.p0(n21) - es);
                        thtesg(n21) = (tg(n21) * deng_exner_pow(1.E5f / C.p0(n21), qese))
                                      * deng_bolton(tg(n21), qese);
                        float dz2 = (n2 == k) ? (z0(klcl) - zlcl1) : dza(n2);
                        float bex = ((2.0f * thtudl) / (thtesg(n21) + thtesg(n2)) - 1.0f) * dz2;
                        if (bex > 0.0f) abeg = abeg + bex * g_;
                    }
                    /* how much CAPE was removed, :2416-2535 */
                    bool decided = false;
                    for (int arm = 0; arm < 4 && !decided; ++arm) {
                        bool hit;
                        if (arm == 0) {
                            hit = (abeg == 0.0f);
                            if (hit) {
                                ainc = ainc * 0.5f;
                                if (ncount < 10) {
                                    next = 255;
                                    decided = true;
                                    break;
                                }
                            }
                        } else if (arm == 1) {
                            float dabe0 = fmaxf(abe - abeg, 0.1f * abe);
                            (void)dabe0;
                            fabe = abeg / (abe + 1.E-8f);
                            hit = (ainc / aincmx > 0.999f && fabe > 1.05f - stab);
                        } else if (arm == 2) {
                            hit = (ncount > 10);
                        } else {
                            hit = (fabe <= 1.05f - stab && fabe >= 0.95f - stab);
                        }
                        if (!hit) continue;
                        if (arm != 0) {
                            /* AINC4 = AINC (never read) */
                        }
                        if (clddpth >= 4.E3f) {
                            kflag = 4;
                            next = 265;
                            decided = true;
                        } else if (clddpth < 4.E3f && (clddpth + zlcl) > zlfc) {
                            float c1 = (clddpth + zlcl) - zlfc;
                            float c2 = (4000.0f - zlfc) + zlcl;
                            if (fabsf(c2) < 0.001f) {
                                *status |= DENG_STATUS_STOP90X;
                                return;
                            }
                            float c3 = c1 / c2;
                            float val = 0.0f;
                            for (int ii = 1; ii <= nt; ++ii)
                                val = val + ainckfsa[(size_t)(ii - 1) * (size_t)ainc_stride];
                            ainc = (ainc + val) / (float)(nt + 1);
                            *ainckfi = ainc;
                            float ainc3 = c3 * ainc + (1.0f - c3) * ainc2;
                            ainc = ainc3;
                            kflag = 3;
                            next = 255;
                            decided = true;
                        }
                    }
                    if (decided) break;
                    float dabe = fmaxf(abe - abeg, 0.1f * abe);
                    ainc = ((ainc * stab) * abe) / (dabe + 1.E-8f);
                    next = 255;
                } while (0);
                if (next == 265) break;
            }
            enter255 = false;
            /* label 255 */
            ainc = fminf(aincmx, ainc);
            const float aincmin = 0.0f;
            ainc = fmaxf(ainc, aincmin);
            pptmlt = pptml2 * ainc;
            tder = tder2 * ainc;
            pptflx = pptfl2 * ainc;
            for (int n2 = 1; n2 <= ltop; ++n2) {
                umf(n2) = umf2(n2) * ainc;
                dmf(n2) = dmf2(n2) * ainc;
                detlq(n2) = detlq2(n2) * ainc;
                detic(n2) = detic2(n2) * ainc;
                udr(n2) = udr2(n2) * ainc;
                uer(n2) = uer2(n2) * ainc;
                der(n2) = der2(n2) * ainc;
                ddr(n2) = ddr2(n2) * ainc;
                dtfm(n2) = 0.0f;
            }
            float dmfmin = uer(lfs) - ems(lfs) / timec;
            if (der(lfs) < dmfmin && dmfmin < 0.0f) {
                float rf = dmfmin / der(lfs);
                for (int n2 = ldb; n2 <= lfs; ++n2) {
                    der2(n2) = der2(n2) * rf;
                    ddr2(n2) = ddr2(n2) * rf;
                    dmf2(n2) = dmf2(n2) * rf;
                    der(n2) = der2(n2) * ainc;
                    ddr(n2) = ddr2(n2) * ainc;
                    dmf(n2) = dmf2(n2) * ainc;
                }
                tder2 = tder2 * rf;
                tder = tder2 * ainc;
                float updin2;
                if (lfs >= klcl) {
                    updin2 = 1.0f - (((1.0f - eqfrc(lfs)) * dmf(lfs)) * updinc) / umf(lfs);
                } else {
                    updin2 = 1.0f;
                }
                cpr = trppt + ppr * (updin2 - 1.0f);
                peff = 1.0f - (tder + ((1.0f - eqfrc(lfs)) * dmf(lfs)) * (rliq(lfs) + rice(lfs)))
                              / (cpr * ainc);
                pptfl2 = peff * cpr;
                pptflx = pptfl2 * ainc;
                float f1 = updin2 / (ainc * updinc);
                for (int n2 = lc; n2 <= lfs; ++n2) {
                    umf2(n2) = umf(n2) * f1;
                    umf(n2) = umf2(n2) * ainc;
                    udr2(n2) = udr(n2) * f1;
                    udr(n2) = udr2(n2) * ainc;
                    uer2(n2) = uer(n2) * f1;
                    uer(n2) = uer2(n2) * ainc;
                    detlq2(n2) = detlq(n2) * f1;
                    detlq(n2) = detlq2(n2) * ainc;
                    detic2(n2) = detic(n2) * f1;
                    detic(n2) = detic2(n2) * ainc;
                    pptml2 = pptml2 - pptice(n2) * (1.0f - updin2 / updinc);
                    pptliq(n2) = (pptliq(n2) * f1) * ainc;
                    pptice(n2) = (pptice(n2) * f1) * ainc;
                }
                pptmlt = pptml2 * ainc;
                updinc = updin2;
            }
            /* GO TO 175 */
        }
        /* label 265: tendencies, :2774-2834 */
        (void)pptmlt;
        for (int kk = kx; kk >= 1; --kk) {
            dtdt(kk) = (tg(kk) - C.t0(kk)) / timec;
            dqdt(kk) = (qqg(kk) - q0(kk)) / timec;
            dqldt(kk) = (qcg(kk) - 0.0f) / timec;
            ddcadt(kk) = (dca(kk) - 0.0f) / timec;
            if (dqldt(kk) <= 1.0E-17f || rh0(kk) > RHCRIT) ddcadt(kk) = 0.0f;
            float area = fmaxf(C.cldareab(kk) + ddcadt(kk) * dt, 0.0f);
            if (area <= 1.0E-17f || dqldt(kk) <= 1.0E-17f || rh0(kk) > RHCRIT) {
                dqcdcdt(kk) = 0.0f;
            } else {
                dqcdcdt(kk) = (dqldt(kk) - C.cldliqb(kk) * ddcadt(kk)) / area;
            }
        }
        *rainshv = ((0.5f * dt2) * pptflx) / dxsq;
        for (int kk = 1; kk <= kx; ++kk) {
            C.tten(kk) = C.tten(kk) + dtdt(kk);
            C.qvten(kk) = C.qvten(kk) + dqdt(kk);
            C.qrten(kk) = C.qrten(kk) + 0.0f;           /* DQRDT */
            if (rh0(kk) > RHCRIT) {
                C.qcten(kk) = C.qcten(kk) + dqldt(kk);
            } else {
                C.dcaten(kk) = C.dcaten(kk) + ddcadt(kk);
                C.qcdcten(kk) = C.qcdcten(kk) + dqcdcdt(kk);
            }
        }
        *rainsh = *rainsh + *rainshv;
    } while (0);

    /* ------------------------------------------------------- label 425 */
    int kdcb = 1, kdct = 1;
    int k1000 = kdct;
    for (int k = kl - 1; k >= 2; --k) {
        if (C.cldareab(k) > 0.001f || rh0(k) > RHCRIT) {
            kdct = k;
            break;
        }
    }
    for (int k = 2; k <= kl - 1; ++k) {
        if (C.cldareab(k) > 0.001f || rh0(k) > RHCRIT) {
            kdcb = k;
            break;
        }
    }
    *kdcldtop = kdct;
    *kdcldbas = kdcb;
    float nupdraft = (ainc != 999.0f) ? ainc : 0.0f;

    if (kdct > 1) {
        const float dtn = dt;
        int kamax = kdcb, klcmax = kdcb;
        for (int k = kdcb; k <= kdct; ++k) {
            if (C.cldareab(k) > C.cldareab(kamax)) kamax = k;
            if (C.cldliqb(k) > C.cldliqb(klcmax)) klcmax = k;
        }
        for (int k = kamax; k >= 1; --k) {
            if (z0(kamax) - z0(k) >= 1000.0f) {
                k1000 = k;
                break;
            }
        }
        k1000 = max(k1000, kdcb);
        for (int k = 1; k <= kl; ++k) {
            if (rh0(k) > RHCRIT) {
                float clq = fmaxf(C.cldliqb(k) + C.qcdcten(k) * dtn, 0.0f);
                float area = fmaxf(C.cldareab(k) + C.dcaten(k) * dtn, 0.0f);
                C.qcdcten(k) = C.qcdcten(k) - clq / dtn;
                C.dcaten(k) = C.dcaten(k) - area / dtn;
                C.qcten(k) = C.qcten(k) + (area * clq) / dtn;
            }
        }
        /* NBC evaporation at the cloud edge, :2975-2996 */
        const float diffk = 1.0E-5f;
        for (int k = 1; k <= kx; ++k) {
            float rlc = fmaxf(C.cldliqb(k) + C.qcdcten(k) * dtn, 0.0f);
            float val = qes(k) - q0(k);
            val = fmaxf(0.0f, val);
            float e2 = (((0.01f + C.cldareab(k)) * diffk) * val) * sqrtf(nupdraft);
            if (rlc <= 1.0E-17f) continue;
            float eovlc = fminf(e2 / rlc, C.cldareab(k) / dtn);
            C.dcaten(k) = C.dcaten(k) - eovlc;
            C.qvten(k) = C.qvten(k) + eovlc * rlc;
            float rlx = (C.t0(k) <= DENG_TO) ? DENG_XLS : (DENG_XLV0 - DENG_XLV1 * C.t0(k));
            float rcp = cp_ * (1.0f + 0.887f * q0(k));
            float rlovcp = rlx / rcp;
            C.tten(k) = C.tten(k) - (eovlc * rlc) * rlovcp;
        }
        /* vertical diffusion of NBC water, :3005-3123 */
        for (int k = kdcb; k <= kdct + 1; ++k) kv(k) = C.kth0(k);
        {
            int k = kamax;
            int kp1 = k + 1;
            int kradlmax = kdcb;
            {
                float x = C.radl0(kdcb);
                for (int kk = kdcb + 1; kk <= kdct; ++kk) {
                    if (C.radl0(kk) < x) {
                        x = C.radl0(kk);
                        kradlmax = kk;
                    }
                }
            }
            int kradsmax = kdcb;
            {
                float x = C.rads0(kdcb);
                for (int kk = kdcb; kk <= kdct; ++kk) {
                    float xar = C.rads0(kk);
                    if (xar >= x) {
                        x = xar;
                        kradsmax = kk;
                    }
                }
            }
            kr(kp1) = ((C.bbls0(k) * C.bbls0(k))
                       / (C.t0(k) * gfk_pow(P00 / C.p0(k), rocpq)))
                      * ((fabsf(C.radl0(kradlmax)) * C.dzq(k)) / 15.0f
                         + (fabsf(C.rads0(kradsmax)) * C.dzq(k)) / 50.0f);
        }
        for (int k = k1000; k <= kamax + 1; ++k) {
            if (kdct + 1 != k1000)
                kr(k) = kr(kamax + 1) * ((float)(k - k1000) / (float)(kamax + 1 - k1000));
        }
        for (int k = kamax + 1; k <= kdct + 1; ++k) kr(k) = kr(kamax + 1);
        for (int k = 1; k <= kdcb - 1; ++k) lflux(k) = 0.0f;
        for (int k = kdct + 1; k <= kxp1; ++k) lflux(k) = 0.0f;
        for (int k = kdct; k >= kdcb; --k) {
            float c1 = (k == kdcb) ? 15.0f : dza(k - 1);
            float area1 = fmaxf(C.cldareab(k) + C.dcaten(k) * dtn, 0.0f);
            float area2 = fmaxf(C.cldareab(k - 1) + C.dcaten(k - 1) * dtn, 0.0f);
            float clq1 = fmaxf(C.cldliqb(k) + C.qcdcten(k) * dtn, 0.0f);
            float clq2 = fmaxf(C.cldliqb(k - 1) + C.qcdcten(k - 1) * dtn, 0.0f);
            float lflux1 = -(((C.rhoe(k) * kv(k)) * (area1 * clq1 - area2 * clq2)) / c1);
            float lflux2 = ((C.rhoe(k) * kr(k)) * (area1 * clq1 - area2 * clq2)) / c1;
            lflux1 = fminf(lflux1, 0.0f);
            lflux2 = fminf(lflux2, 0.0f);
            lflux(k) = lflux1 + lflux2;
            c1 = lflux(k + 1) - ((clq1 * dp(k)) / (g_ * dtn)) * 0.05f;
            lflux(k) = fmaxf(lflux(k), c1);
        }
        for (int k = kdct + 1; k >= kdcb; --k) {
            if (rh0(k) > RHCRIT || (C.cldareab(k) + C.dcaten(k) * dtn) <= 1.0e-6f) {
                lflux(k) = 0.0f;
                lflux(k + 1) = 0.0f;
            }
        }
        float rhcr = 0.95f;
        if (q0(kdcb - 1) / qes(kdcb - 1) > rhcr) {
            lflux(kdcb) = 0.0f;
        } else {
            float val1 = -((g_ * (lflux(kdcb) - lflux(kdcb - 1))) / dp(kdcb - 1));
            float val2 = (rhcr * qes(kdcb - 1) - q0(kdcb - 1)) / dtn;
            float val = fminf(val1, val2);
            lflux(kdcb) = -((val * dp(kdcb - 1)) / g_);
        }
        for (int k = kdct; k >= kdcb; --k) {
            float area = fmaxf(C.cldareab(k) + C.dcaten(k) * dtn, 0.0f);
            if (rh0(k) > RHCRIT || area <= 1.0e-6f) continue;
            C.qcdcten(k) = C.qcdcten(k) - ((g_ * (lflux(k + 1) - lflux(k))) / dp(k)) / area;
        }
        {
            int k = kdcb - 1;
            float rlx = (C.t0(k) <= DENG_TO) ? DENG_XLS : (DENG_XLV0 - DENG_XLV1 * C.t0(k));
            float rcp = cp_ * (1.0f + 0.887f * q0(k));
            float rlovcp = rlx / rcp;
            C.qvten(k) = C.qvten(k) - (g_ * (lflux(k + 1) - lflux(k))) / dp(k);
            C.tten(k) = C.tten(k) + ((g_ * (lflux(k + 1) - lflux(k))) / dp(k)) * rlovcp;
        }
        /* exposed cloud base, :3141-3186 */
        for (int k = kdct; k >= kdcb + 1; --k) {
            float areadiff = C.cldareab(k) - C.cldareab(k - 1);
            areadiff = fmaxf(areadiff, 0.0f);
            float clq = fmaxf(C.cldliqb(k) + C.qcdcten(k) * dtn, 0.0f);
            float c1 = -(((clq * dp(k)) / (g_ * dtn)) * 0.05f);
            float fx1 = -((((C.rhoe(k) * kv(k)) * C.cldliqb(k)) / 15.0f) * areadiff);
            float fx2 = -((((C.rhoe(k) * kr(k)) * C.cldliqb(k)) / 15.0f) * areadiff);
            float fx = fx1 + fx2;
            fx = fmaxf(fx, c1);
            if ((C.cldareab(k) + C.dcaten(k) * dtn) <= 1.0E-6f) continue;
            if (rh0(k) > RHCRIT) continue;
            rhcr = 0.95f;
            if (q0(k - 1) / qes(k - 1) > rhcr) {
                fx = 0.0f;
            } else {
                float val1 = -((g_ * fx) / dp(k - 1));
                float val2 = (rhcr * qes(k - 1) - q0(k - 1)) / dtn;
                float val = fminf(val1, val2);
                fx = -((val * dp(k - 1)) / g_);
            }
            float area = fmaxf(C.cldareab(k) + C.dcaten(k) * dtn, 0.0f);
            if (area <= 1.0E-6f) continue;
            C.qcdcten(k) = C.qcdcten(k) + ((g_ * fx) / dp(k)) / area;
            float rlx = (C.t0(k - 1) <= DENG_TO) ? DENG_XLS : (DENG_XLV0 - DENG_XLV1 * C.t0(k - 1));
            float rcp = cp_ * (1.0f + 0.887f * q0(k - 1));
            float rlovcp = rlx / rcp;
            C.qvten(k - 1) = C.qvten(k - 1) - (g_ * fx) / dp(k - 1);
            C.tten(k - 1) = C.tten(k - 1) + ((g_ * fx) / dp(k - 1)) * rlovcp;
        }
        /* cloud-top entrainment instability, :3191-3299 */
        do {
            if (rh0(kdct) > RHCRIT) break;
            const float efold = 1.0E-4f, ex = 1.0f, depthmax = 100.0f;
            if (kdct <= 0) break;
            float hin, hout;
            if (C.t0(kdct) >= DENG_TO) {
                hin = (cp_ * C.t0(kdct) + g_ * z0(kdct)) + DENG_XLV * qes(kdct);
            } else {
                hin = (cp_ * C.t0(kdct) + g_ * z0(kdct)) + DENG_XLS * qes(kdct);
            }
            if (C.t0(kdct + 1) >= DENG_TO) {
                hout = (cp_ * C.t0(kdct + 1) + g_ * z0(kdct + 1)) + DENG_XLV * q0(kdct + 1);
            } else {
                hout = (cp_ * C.t0(kdct + 1) + g_ * z0(kdct + 1)) + DENG_XLS * q0(kdct + 1);
            }
            float delth = hin - hout;
            float qin = qes(kdct) + C.cldliqb(kdct);
            float qout = q0(kdct + 1);
            float deltq = qin - qout;
            float ratio = delth / (DENG_XLV * deltq + 1.E-8f);
            float rkapa = ((cp_ * 0.5f) * (C.t0(kdct) + C.t0(kdct + 1))) / DENG_XLV;
            const float delta = 0.608f;
            float dqssdt, rlx;
            if (C.t0(kdct) > DENG_TO) {
                dqssdt = ((qes(kdct) * DENG_SVP2) * (DENG_SVPT0 - DENG_SVP3))
                         / ((C.t0(kdct) - DENG_SVP3) * (C.t0(kdct) - DENG_SVP3));
                rlx = DENG_XLV0 - DENG_XLV1 * C.t0(kdct);
            } else {
                dqssdt = (qes(kdct) * 6.15E3f) / (C.t0(kdct) * C.t0(kdct));
                rlx = DENG_XLS;
            }
            float rcp = cp_ * (1.0f + 0.887f * q0(kdct));
            float rlovcp = rlx / rcp;
            float gama = rlovcp * dqssdt;
            float beita = (1.0f + ((1.0f + delta) * gama) * rkapa) / (1.0f + gama);
            float ratiomin = rkapa / beita;
            float ratiomax = ((1.0f + gama) * (1.0f + (1.0f - delta) * rkapa))
                             / (2.0f + (1.0f + (1.0f + delta) * rkapa) * gama);
            float sigmma;
            if (ratio <= ratiomin) {
                break;                                     /* SIGMMA = 0, GOTO 340 */
            } else if (ratio >= ratiomax) {
                sigmma = efold;
            } else {
                sigmma = efold * gfk_pow((ratio - ratiomin) / (ratiomax - ratiomin), ex);
            }
            int k100 = kdct;
            for (int l = kdct - 1; l >= kdcb; --l) {
                if ((z0(kdct) + 0.5f * C.dzq(kdct)) - (z0(l) - 0.5f * C.dzq(l)) >= depthmax) {
                    k100 = k100 - 1;
                    break;
                }
                k100 = l;
            }
            rhcr = 0.95f;
            if (rh0(kdct + 1) > rhcr) break;
            for (int l = kdct; l >= k100; --l) {
                if (rh0(l) > RHCRIT) continue;
                float wfun = (float)(l - k100 + 1) / (float)(kdct - k100 + 1);
                float area = fmaxf(C.cldareab(l) + C.dcaten(l) * dtn, 0.0f);
                float rly = (C.t0(l) <= DENG_TO) ? DENG_XLS : (DENG_XLV0 - DENG_XLV1 * C.t0(l));
                float rcpl = cp_ * (1.0f + 0.887f * q0(l));
                float rlovcpl = rly / rcpl;
                float val1 = (sigmma * C.cldliqb(l)) * wfun;
                C.qcdcten(l) = C.qcdcten(l) - val1;
                C.qvten(kdct + 1) = C.qvten(kdct + 1) + ((val1 * area) * dp(l)) / dp(kdct + 1);
                C.tten(l) = C.tten(l) - (val1 * area) * rlovcpl;
                float val = q0(kdct + 1) + C.qvten(kdct + 1) * dtn;
                if (val / qes(kdct + 1) > rhcr) {
                    C.qvten(kdct + 1) = C.qvten(kdct + 1) - (val1 * dp(l)) / dp(kdct + 1);
                    C.tten(l) = C.tten(l) + (val1 * area) * rlovcpl;
                }
            }
        } while (0);
        /* label 340: drizzle, :3314-3370 */
        {
            const float bvts3 = 3.0f + BVTS;
            const float ppi = 1.0f / (PIE * N0R);
            const float g3pbs = deng_gamma(3.0f + BVTS, status);
            const float pracs = ((((PIE * N0S) * AVTS) * g3pbs) * 0.25f) * ESI;
            for (int k = 1; k <= kx; ++k) {
                float tout = C.t0(k);
                float sc2 = fmaxf(0.0f, C.cldliqb(k));
                float sc3 = fmaxf(0.0f, C.qr0(k));
                float sc7;
                if (tout > DENG_TO) {
                    sc7 = gfk_pow(((C.rhoe(k) * sc3) * ppi) / 1000.0f, 0.25f);
                } else {
                    float rhos = 0.1f;
                    float ppis = 1.0f / ((PIE * N0S) * rhos);
                    sc7 = gfk_pow(((C.rhoe(k) * sc3) * ppis) / 1000.0f, 0.25f);
                }
                float prc, pra;
                if (tout < DENG_TO) {
                    float xnc = (XN0 * gfk_exp(0.6f * (DENG_TO - tout))) / C.rhoe(k);
                    xnc = fmaxf(xnc, 10000.0f / C.rhoe(k));
                    prc = fmaxf(0.0f, (sc2 - XMMAX * xnc) / dtn);
                    if (sc3 < 1.0E-17f) {
                        pra = 0.0f;
                    } else {
                        pra = (pracs * gfk_pow(sc7, bvts3)) * sc2;
                    }
                } else {
                    prc = fmaxf(0.0f, QCK1 * (sc2 - QCTH));
                    float bvt3 = 3.0f + BVT;
                    if (sc3 < 1.0E-17f) {
                        pra = 0.0f;
                    } else {
                        float g3pb = deng_gamma(3.0f + BVT, status);
                        float prac = (((PIE * N0R) * AVT) * g3pb) * 0.25f;
                        pra = (prac * gfk_pow(sc7, bvt3)) * sc2;
                    }
                }
                float val = prc + pra;
                val = fminf(val, C.cldliqb(k) / dtn);
                float area = fmaxf(C.cldareab(k) + C.dcaten(k) * dtn, 0.0f);
                if (area < 1.0E-17f) val = 0.0f;
                C.qrten(k) = C.qrten(k) + val * area;
                C.qcdcten(k) = C.qcdcten(k) - val;
            }
        }
        /* ice sedimentation, :3386-3440 */
        {
            for (int k = 1; k <= kl; ++k) {
                float clq = fmaxf(C.cldliqb(k) + C.qcdcten(k) * dtn, 0.0f);
                float area = fmaxf(C.cldareab(k) + C.dcaten(k) * dtn, 0.0f);
                scr3(k) = clq * area;
            }
            int nstep = 1;
            for (int k = 1; k <= kl; ++k) {
                float rho2 = C.p0(k) / (r_ * (C.t0(k) + dtn * C.tten(k)));
                float vt2c;
                if ((C.t0(k) + dtn * C.tten(k)) > DENG_TO) {
                    vt2c = 0.0f;
                } else {
                    vt2c = 3.29f * gfk_pow(rho2 * scr3(k), 0.16f);
                }
                rgvc(k) = (g_ * rho2) * vt2c;
                nstep = max((int)(((rgvc(k) * dtn) / C.dsigma[k - 1]) + 1.0f), nstep);
            }
            if (nstep > 1000) {
                *status |= DENG_STATUS_SETTLING;
                return;
            }
            const float fnstep = (float)nstep;
            for (int n = 1; n <= nstep; ++n) {
                for (int k = 1; k <= kl; ++k) faloutc(k) = rgvc(k) * scr3(k);
                for (int k = kl - 1; k >= 2; --k) {
                    float area = fmaxf(C.cldareab(k) + C.dcaten(k) * dtn, 0.0f);
                    float clq = fmaxf(C.cldliqb(k) + C.qcdcten(k) * dtn, 0.0f);
                    if (rh0(k) > RHCRIT) continue;
                    if (area <= 1.E-4f) continue;
                    float faltndc = (faloutc(k) - faloutc(k + 1)) / C.dsigma[k - 1];
                    float c1 = fminf(faltndc, ((area * clq) / dtn) * fnstep);
                    c1 = fmaxf(0.0f, c1);
                    C.qcdcten(k) = C.qcdcten(k) - (c1 / area) / fnstep;
                    scr3(k) = scr3(k) - ((c1 * dtn) / fnstep) / area;
                    C.qcten(k - 1) = C.qcten(k - 1)
                        + ((c1 * C.dsigma[k - 1]) / C.dsigma[k - 2]) / fnstep;
                    rgvc(k) = fmaxf(rgvc(k) / C.dsigma[k - 1], rgvc(k + 1) / C.dsigma[k])
                              * C.dsigma[k - 1];
                }
            }
        }
    }
    /* label 345 */
    *clddpthb = clddpth;
    *cldtopb = cldtop;
    *ltopb = ltop;
    for (int k = 1; k <= kl; ++k) C.wub(k) = wu(k);
}

/* ==================================================== deng_shcu_driver
 * One column: module_shcu_deng.F:107-424 at this column, then
 * module_shallowcu_driver.F:686-691.  Inputs (nz, ncol), column fastest;
 * Z_AT_W and W (nz+1, ncol); AINCKFSA (100, ncol).  DX is this column's
 * grid length and DNW its DSIGMA(1:nz).  Both entry points below call this
 * body; deng_shcu_column (WRF) hands it the domain's DX and DNW. */
__device__ __forceinline__ void deng_column_body(
        const float* __restrict__ u, const float* __restrict__ v,
        const float* __restrict__ t, const float* __restrict__ qv,
        const float* __restrict__ qc, const float* __restrict__ qr,
        const float* __restrict__ p, const float* __restrict__ pii,
        const float* __restrict__ rho, const float* __restrict__ dz8w,
        const float* __restrict__ tke_in, const float* __restrict__ exch_h,
        const float* __restrict__ el_pbl, const float* __restrict__ radl,
        const float* __restrict__ rads, const float* __restrict__ z_at_w,
        const float* __restrict__ w, const float* __restrict__ pblh,
        const float* __restrict__ htopo, const float* __restrict__ xlong,
        const float* __restrict__ dnw, const float* __restrict__ pratesh,
        float* w0avg, float* tkeavg, float* cldareaa, float* cldareab,
        float* cldliqa, float* cldliqb, float* wub,
        float* rainsh, float* rainshv, float* rainshvb, float* pblmax,
        float* capesave, float* xtime1, float* radsave, float* clddpthb,
        float* cldtopb, float* mavail, float* pblhavg,
        int* ltopb, int* kdcldtop, int* kdcldbas, float* ainckfsa,
        float* rthshten, float* rqvshten, float* rqcshten, float* rqrshten,
        float* rushten, float* rvshten, float* rdcashten, float* rqcdcshten,
        float* cldfra_sh, float* ca_rad, float* cw_rad, int* status_out,
        float* ws, float dt, float dx, float xtime, float gmt, int adapt,
        int tke_from_qke, int nz, int ncol, int column) {
    if (nz < 3 || nz > DENG_KMAX) return;
    const int kp = nz + 1;
    float* wsb = DENGWS_LANE_BASE(ws, kp);
    for (int s = 0; s < DENGWS_SLOTS; ++s) {
        DCol c = DWS(wsb, s, kp);
        for (int k = 1; k <= kp; ++k) c(k) = 0.0f;
    }
    int status = 0;
    const DIn T{t, ncol, column}, QV{qv, ncol, column}, QC{qc, ncol, column};
    const DIn P{p, ncol, column}, PI{pii, ncol, column}, W{w, ncol, column};
    const DIn TKE{tke_in, ncol, column};
    const DOut W0AVG{w0avg, ncol, column}, TKEAVG{tkeavg, ncol, column};
    const DOut CLDAREAA{cldareaa, ncol, column}, CLDAREAB{cldareab, ncol, column};
    const DOut CLDLIQA{cldliqa, ncol, column}, CLDLIQB{cldliqb, ncol, column};
    const DOut WUB{wub, ncol, column};
    const DOut RTHSHTEN{rthshten, ncol, column}, RQVSHTEN{rqvshten, ncol, column};
    const DOut RQCSHTEN{rqcshten, ncol, column}, RQRSHTEN{rqrshten, ncol, column};
    const DOut RUSHTEN{rushten, ncol, column}, RVSHTEN{rvshten, ncol, column};
    const DOut RDCASHTEN{rdcashten, ncol, column}, RQCDCSHTEN{rqcdcshten, ncol, column};
    const DOut CLDFRA_SH{cldfra_sh, ncol, column}, CA_RAD{ca_rad, ncol, column};
    const DOut CW_RAD{cw_rad, ncol, column};

    const float dxsq = dx * dx;
    const float dt2 = 2.0f * dt;
    const float gnuhf = 0.1f;
    const float omuhf = 1.0f - 2.0f * gnuhf;

    /* W time average on the half levels (if_avg_w = 1), :133-153 and
     * time_avg_3d :3723-3765 */
    {
        const float tpam = 10.0f;
        int ntst = deng_nint((tpam * 60.0f) / dt);
        float tst = (float)ntst;
        float avgfctr, fctr, den;
        if (adapt) {
            avgfctr = fmaxf(tpam * 60.0f, dt) - dt;
            fctr = dt;
            den = fmaxf(tpam * 60.0f, dt);
        } else {
            avgfctr = tst - 1.0f;
            fctr = 1.0f;
            den = tst;
        }
        for (int k = 1; k <= nz; ++k) {
            float wh = 0.5f * (W(k) + W(k + 1));
            W0AVG(k) = (W0AVG(k) * avgfctr + wh * fctr) / den;
        }
    }
    /* PBLH and TKE "averages" (if_avg_pbl = if_avg_tke = 0): the current
     * values; TKE by the shallowcu_driver's bl_pbl_physics selection. */
    pblhavg[column] = pblh[column];
    for (int k = 1; k <= nz; ++k)
        TKEAVG(k) = tke_from_qke ? 0.5f * TKE(k) : TKE(k);

    /* the RH-based cloud fraction for radiation, :235-263 */
    DCol rh = DWS(wsb, 1, kp);
    for (int k = 1; k <= nz; ++k) {
        float tk = T(k);
        float es;
        if (tk >= DENG_TO) {
            es = deng_esliq(tk);
        } else {
            es = deng_esice(tk);
        }
        float qes = (0.622f * es) / (P(k) - es);
        rh(k) = fmaxf(QV(k) / qes, 1.0E-25f);
        const float alf1 = 0.25f, alf2 = 0.49f, alf3 = 100.0f;
        float cs;
        if (rh(k) < 0.999f) {
            float c1 = (1.0f - rh(k)) * gfk_pow(qes, alf2);
            float qctot = CLDLIQA(k) * CLDAREAA(k) + QC(k);
            float c2 = -((alf3 * qctot) / c1);
            cs = gfk_pow(rh(k), alf1) * (1.0f - gfk_exp(c2));
        } else {
            cs = 1.0f;
        }
        CA_RAD(k) = (1.0f - CLDAREAA(k)) * cs + CLDAREAA(k);
        CLDFRA_SH(k) = CA_RAD(k);
        CW_RAD(k) = CLDLIQA(k) * CLDAREAA(k);
    }

    /* the column call, :265-342 */
    DengColumn C;
    C.u0 = DIn{u, ncol, column};
    C.v0 = DIn{v, ncol, column};
    C.t0 = T;
    C.qr0 = DIn{qr, ncol, column};
    C.p0 = P;
    C.rhoe = DIn{rho, ncol, column};
    C.dzq = DIn{dz8w, ncol, column};
    C.kth0 = DIn{exch_h, ncol, column};
    C.bbls0 = DIn{el_pbl, ncol, column};
    C.radl0 = DIn{radl, ncol, column};
    C.rads0 = DIn{rads, ncol, column};
    C.zw0 = DIn{z_at_w, ncol, column};
    C.w0avg0 = W0AVG;
    C.tke0 = TKEAVG;
    C.dsigma = dnw;
    C.rh0 = rh;
    C.q0 = DWS(wsb, 0, kp);
    C.cldareab = CLDAREAB;
    C.cldliqb = CLDLIQB;
    C.wub = WUB;
    C.qvten = DWS(wsb, 77, kp);
    C.qcten = DWS(wsb, 78, kp);
    C.qrten = DWS(wsb, 79, kp);
    C.tten = DWS(wsb, 80, kp);
    C.dcaten = DWS(wsb, 81, kp);
    C.qcdcten = DWS(wsb, 82, kp);
    for (int k = 1; k <= nz; ++k) {
        C.qvten(k) = 0.0f; C.qcten(k) = 0.0f; C.qrten(k) = 0.0f;
        C.tten(k) = 0.0f; C.dcaten(k) = 0.0f; C.qcdcten(k) = 0.0f;
        C.q0(k) = fmaxf(QV(k), 1.E-12f);
    }
    float rshv = 0.0f, rsh = 0.0f;
    float pmax = pblmax[column];
    float ctopb = cldtopb[column];
    float cdepb = clddpthb[column];
    int ltb = ltopb[column];
    int kdt = kdcldtop[column], kdb = kdcldbas[column];
    float capei = 0.0f, radiusc = 0.0f, ainckfi = 0.0f;
    deng_shcu(C, wsb, kp, nz, dt, dt2, dx, dxsq, xtime, gmt,
              pblhavg[column], htopo[column], xlong[column],
              capesave[column], radsave[column], ainckfsa + column, ncol,
              &rshv, &rsh, &pmax, &ctopb, &cdepb, &ltb, &kdt, &kdb,
              &capei, &radiusc, &ainckfi, &status);
    status_out[column] = status;
    if (status & (DENG_STATUS_STOP907 | DENG_STATUS_STOP90X
                  | DENG_STATUS_SETTLING | DENG_STATUS_LCL_AT_SURFACE
                  | DENG_STATUS_NO_500HPA)) {
        return;
    }
    pblmax[column] = pmax;
    cldtopb[column] = ctopb;
    clddpthb[column] = cdepb;
    ltopb[column] = ltb;
    kdcldtop[column] = kdt;
    kdcldbas[column] = kdb;

    for (int k = 1; k <= nz; ++k) {
        RUSHTEN(k) = 0.0f;
        RVSHTEN(k) = 0.0f;
        RTHSHTEN(k) = (C.tten(k) / PI(k)) * 1.0f;
        RQVSHTEN(k) = C.qvten(k) * 1.0f;
        RQCSHTEN(k) = C.qcten(k) * 1.0f;
        RQRSHTEN(k) = C.qrten(k) * 1.0f;
        RDCASHTEN(k) = C.dcaten(k) * 1.0f;
        RQCDCSHTEN(k) = C.qcdcten(k) * 1.0f;
    }
    /* surface moisture availability, :361-370 */
    if (rshv == 0.0f && rainshvb[column] != 0.0f) xtime1[column] = xtime + 60.0f;
    float amois = mavail[column];
    if (rshv != 0.0f || xtime <= xtime1[column]) {
        mavail[column] = amois + 0.67f * (1.0f - amois);
    } else {
        mavail[column] = amois;
    }
    rainshvb[column] = rshv;
    for (int ii = 100; ii >= 2; --ii)
        ainckfsa[(size_t)(ii - 1) * (size_t)ncol + column] =
            ainckfsa[(size_t)(ii - 2) * (size_t)ncol + column];
    radsave[column] = radiusc;
    ainckfsa[column] = ainckfi;
    capesave[column] = capei;

    /* NBC leapfrog, :385-424 */
    for (int k = 1; k <= nz; ++k) {
        float areac = CLDAREAB(k) + RDCASHTEN(k) * dt;
        float liqc = CLDLIQB(k) + RQCDCSHTEN(k) * dt;
        areac = fmaxf(0.0f, areac);
        liqc = fmaxf(0.0f, liqc);
        if (areac > 0.99999f) {
            liqc = liqc * areac;
            areac = 1.0f;
        }
        if (areac <= 1.0e-17f || liqc <= 1.0e-17f) {
            areac = 0.0f;
            liqc = 0.0f;
        }
        CLDAREAB(k) = omuhf * CLDAREAA(k) + gnuhf * (CLDAREAB(k) + areac);
        CLDLIQB(k) = omuhf * CLDLIQA(k) + gnuhf * (CLDLIQB(k) + liqc);
        CLDAREAA(k) = areac;
        CLDLIQA(k) = liqc;
        if (CLDAREAA(k) <= 1.0e-17f || CLDLIQA(k) <= 1.0e-17f) {
            CLDAREAA(k) = 0.0f;
            CLDLIQA(k) = 0.0f;
        }
        if (cdepb <= 0.0f) {
            RQCSHTEN(k) = RQCSHTEN(k) + (CLDAREAB(k) * CLDLIQB(k)) / dt;
            CLDAREAB(k) = 0.0f;
            CLDLIQB(k) = 0.0f;
        }
    }
    /* deng_shcu_driver leaves RAINSH = RAINSH(=0) + RAINSHV and its own
     * RAINSHV; shallowcu_driver then overwrites RAINSHV with PRATESH*DT,
     * and Deng never writes PRATESH (module_shallowcu_driver.F:686-691). */
    rainsh[column] = rsh;
    rainshv[column] = pratesh[column] * dt;
}

/* WRF: every column of a (nz, ny, nx) tile, the domain's DX and DNW. */
extern "C" __global__ __launch_bounds__(256)
void deng_shcu_column(
        const float* __restrict__ u, const float* __restrict__ v,
        const float* __restrict__ t, const float* __restrict__ qv,
        const float* __restrict__ qc, const float* __restrict__ qr,
        const float* __restrict__ p, const float* __restrict__ pii,
        const float* __restrict__ rho, const float* __restrict__ dz8w,
        const float* __restrict__ tke_in, const float* __restrict__ exch_h,
        const float* __restrict__ el_pbl, const float* __restrict__ radl,
        const float* __restrict__ rads, const float* __restrict__ z_at_w,
        const float* __restrict__ w, const float* __restrict__ pblh,
        const float* __restrict__ htopo, const float* __restrict__ xlong,
        const float* __restrict__ dnw, const float* __restrict__ pratesh,
        float* w0avg, float* tkeavg, float* cldareaa, float* cldareab,
        float* cldliqa, float* cldliqb, float* wub,
        float* rainsh, float* rainshv, float* rainshvb, float* pblmax,
        float* capesave, float* xtime1, float* radsave, float* clddpthb,
        float* cldtopb, float* mavail, float* pblhavg,
        int* ltopb, int* kdcldtop, int* kdcldbas, float* ainckfsa,
        float* rthshten, float* rqvshten, float* rqcshten, float* rqrshten,
        float* rushten, float* rvshten, float* rdcashten, float* rqcdcshten,
        float* cldfra_sh, float* ca_rad, float* cw_rad, int* status_out,
        float* ws, float dt, float dx, float xtime, float gmt, int adapt,
        int tke_from_qke, int nz, int ny, int nx, int col0) {
    const int ncol = ny * nx;
    const int column = col0 + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (column >= ncol) return;
    deng_column_body(u, v, t, qv, qc, qr, p, pii, rho, dz8w, tke_in, exch_h,
        el_pbl, radl, rads, z_at_w, w, pblh, htopo, xlong, dnw, pratesh,
        w0avg, tkeavg, cldareaa, cldareab, cldliqa, cldliqb, wub,
        rainsh, rainshv, rainshvb, pblmax, capesave, xtime1, radsave, clddpthb,
        cldtopb, mavail, pblhavg, ltopb, kdcldtop, kdcldbas, ainckfsa,
        rthshten, rqvshten, rqcshten, rqrshten, rushten, rvshten, rdcashten,
        rqcdcshten, cldfra_sh, ca_rad, cw_rad, status_out,
        ws, dt, dx, xtime, gmt, adapt, tke_from_qke, nz, ncol, column);
}

/* MPAS (woof.hex.mpasphys.shcu_deng_v842, WOOF-only): the same body with a
 * grid length per cell, dxcol (ncol,), and a DSIGMA per cell, dnwcol
 * (ncol, nz) -- the variable-resolution mesh has no single DX and the
 * height coordinate no shared eta.  ny = 1, nx = the owned cells. */
extern "C" __global__ __launch_bounds__(256)
void deng_shcu_mpas_column(
        const float* __restrict__ u, const float* __restrict__ v,
        const float* __restrict__ t, const float* __restrict__ qv,
        const float* __restrict__ qc, const float* __restrict__ qr,
        const float* __restrict__ p, const float* __restrict__ pii,
        const float* __restrict__ rho, const float* __restrict__ dz8w,
        const float* __restrict__ tke_in, const float* __restrict__ exch_h,
        const float* __restrict__ el_pbl, const float* __restrict__ radl,
        const float* __restrict__ rads, const float* __restrict__ z_at_w,
        const float* __restrict__ w, const float* __restrict__ pblh,
        const float* __restrict__ htopo, const float* __restrict__ xlong,
        const float* __restrict__ dnwcol, const float* __restrict__ pratesh,
        float* w0avg, float* tkeavg, float* cldareaa, float* cldareab,
        float* cldliqa, float* cldliqb, float* wub,
        float* rainsh, float* rainshv, float* rainshvb, float* pblmax,
        float* capesave, float* xtime1, float* radsave, float* clddpthb,
        float* cldtopb, float* mavail, float* pblhavg,
        int* ltopb, int* kdcldtop, int* kdcldbas, float* ainckfsa,
        float* rthshten, float* rqvshten, float* rqcshten, float* rqrshten,
        float* rushten, float* rvshten, float* rdcashten, float* rqcdcshten,
        float* cldfra_sh, float* ca_rad, float* cw_rad, int* status_out,
        float* ws, float dt, const float* __restrict__ dxcol, float xtime, float gmt, int adapt,
        int tke_from_qke, int nz, int ny, int nx, int col0) {
    const int ncol = ny * nx;
    const int column = col0 + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (column >= ncol) return;
    deng_column_body(u, v, t, qv, qc, qr, p, pii, rho, dz8w, tke_in, exch_h,
        el_pbl, radl, rads, z_at_w, w, pblh, htopo, xlong, dnwcol + (size_t)column * (size_t)nz, pratesh,
        w0avg, tkeavg, cldareaa, cldareab, cldliqa, cldliqb, wub,
        rainsh, rainshv, rainshvb, pblmax, capesave, xtime1, radsave, clddpthb,
        cldtopb, mavail, pblhavg, ltopb, kdcldtop, kdcldbas, ainckfsa,
        rthshten, rqvshten, rqcshten, rqrshten, rushten, rvshten, rdcashten,
        rqcdcshten, cldfra_sh, ca_rad, cw_rad, status_out,
        ws, dt, dxcol[column], xtime, gmt, adapt, tke_from_qke, nz, ncol, column);
}

/* ============================================ MPAS column words (WOOF-only)
 * woof.hex.mpasphys.shcu_deng_v842: the per-cell quantities the MPAS path
 * hands deng_shcu_mpas_column and the coupling of its tendencies.  Each
 * statement is one binary32 operation in the order written. */

/* DX = len_disp / meshDensity**0.25 (mpas_atmphys_driver_convection.F:718,
 * glibc powf) and XLONG = lonCell / degrad (the MPAS radiation's xlon_p,
 * mpas_atmphys_driver_radiation_sw.F:422), (nc,). */
extern "C" __global__ void deng_mpas_surface(int nc, float len_disp,
        float degrad, const float* __restrict__ density,
        const float* __restrict__ lon, float* __restrict__ dx,
        float* __restrict__ xlong) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= nc) return;
    dx[i] = len_disp / gfk_pow(density[i], 0.25f);
    xlong[i] = lon[i] / degrad;
}

/* DNW per cell, (nc, nl): WRF's eta increment of the column's dry
 * hydrostatic w-level pressure pd (nl+1 rows of ``stride`` cells),
 * (pd(k+1) - pd(k)) / (pd(1) - pd(nl+1)). */
extern "C" __global__ void deng_mpas_dnw(int nl, int nc, int stride,
        const float* __restrict__ pd, float* __restrict__ dnw) {
    const int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= nc) return;
    const float depth = pd[c] - pd[(size_t)nl * stride + c];
    for (int k = 0; k < nl; ++k)
        dnw[(size_t)c * nl + k] =
            (pd[(size_t)(k + 1) * stride + c] - pd[(size_t)k * stride + c]) / depth;
}

/* The Deng tendencies (nl, nc) added to physics_get_tend's sources on the
 * owned cells as one more coupled term in mpas_atmphys_todynamics.F's
 * arithmetic (physics_get_tend_work): th = RTHSHTEN*m, qv = RQVSHTEN*m,
 * coeff = 1 + rvord*qv_state, rtheta += coeff*th + rvord*theta_m*qv/coeff,
 * q(v,c,r) += RQ*SHTEN*m.  State arrays have ``sstride`` cells per level,
 * the sources ``dstride``; ``plane`` is one scalar's source plane. */
extern "C" __global__ void deng_mpas_couple(int nl, int nc, int sstride,
        int dstride, int plane, int iqv, int iqc, int iqr, float rvord,
        const float* __restrict__ mass, const float* __restrict__ theta_m,
        const float* __restrict__ qv, const float* __restrict__ rth,
        const float* __restrict__ rqv, const float* __restrict__ rqc,
        const float* __restrict__ rqr, float* rho_theta, float* scalars) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= nl * nc) return;
    const int k = i / nc, c = i % nc;
    const size_t s = (size_t)k * sstride + c;
    const size_t d = (size_t)k * dstride + c;
    const float m = mass[s];
    const float th = rth[i] * m;
    const float vq = rqv[i] * m;
    const float coeff = 1.0f + rvord * qv[s];
    const float coupled = coeff * th + ((rvord * theta_m[s]) * vq) / coeff;
    rho_theta[d] = rho_theta[d] + coupled;
    scalars[(size_t)iqv * plane + d] = scalars[(size_t)iqv * plane + d] + vq;
    scalars[(size_t)iqc * plane + d] = scalars[(size_t)iqc * plane + d] + rqc[i] * m;
    scalars[(size_t)iqr * plane + d] = scalars[(size_t)iqr * plane + d] + rqr[i] * m;
}
