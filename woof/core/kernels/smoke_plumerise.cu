// The Freitas et al. (2007) 1-D plume-rise model of WRF-Chem v4.7.1,
// chem/module_chem_plumerise_scalar.F (plumerise and everything it calls),
// one thread block per burning column.  Driven by woof/smoke/plume_rise.py.
//
// Every statement follows WRF's own, in WRF's order of evaluation, in
// real(4).  The unit is compiled with -fmad=false (gfortran fuses no
// product into a sum), divisions and sqrtf are IEEE, Fortran's real(4) EXP
// and real-exponent ** are glibc's expf and powf (gfk_exp, gfk_pow from
// glibc_flt32.cuh, which precedes this file in the unit), and integer
// powers are products as gfortran expands them.  Fortran MAX(a, b) is
// a > b ? a : b and MIN(a, b) is a < b ? a : b, as gfortran evaluates them
// (measured: MAX(0., -0.) is -0.).  Divisions by a constant are spelled
// __fdiv_rn, the IEEE quotient on every card.
//
// THE PARALLEL FORM.  The plume model's module variables
// (module_zero_plumegen_coms.F) are the block's shared memory, SP_NARR
// arrays of SP_LEV levels, Fortran's one-based indices.  A level k always
// belongs to thread (k - 1) % blockDim.x, so a thread meets its own levels
// in program order, and a __syncthreads() separates every loop that reads a
// level another thread writes.  WRF's loops over levels are elementwise
// except four, done exactly as WRF does them:
//   * the prefix chains (htint's search, the Exner integral, INITIAL's
//     radius and viscosity) and the scans for the plume top and wmax run on
//     thread 0 (a maximum is exact in any order, but it is taken in WRF's);
//   * fallpart reads VTH(k+1)/VTI(k+1) before the loop reaches k+1: the old
//     words are read first, then the new ones written;
//   * the microphysics loop (makeplume:695-713) carries one dependence,
//     QSAT(L-1) set by the iteration before, and WATERBAL(L) touches level
//     L only, so QSAT, EST and RHO of every level are formed first (from
//     T(L), which nothing before WATERBAL(L) changes), then every level's
//     DQSDZ from QSAT(L+1) as the step left it and the new QSAT(L-1);
//     the module scalars WBAR and DQSDZ keep their L = nm1-1 values, as
//     WRF's loop leaves them, for lbound's WATERBAL at the next step.
// The scalars live in shared memory and only thread 0 writes them.
//
// Every column starts from the module's first-call state (all zero), the
// state a fresh WRF process gives the first column it computes; WRF's later
// columns inherit scalars and levels above the plume grid from the column
// before (tools/smoke_wrf471_oracle/PROVENANCE.md).  The plume
// grid (set_grid: zt = 0, 50, 150, ...; zm = 0, 100, 200, ...) is formed on
// the fly; its values are integers, exact in real(4), so the closed forms
// are set_grid's words.

#define SP_NKP 200
#define SP_LEV (SP_NKP + 3)
#define SP_NARR 44
#define SP_NSTATE 12

enum {
    SP_W, SP_T, SP_QV, SP_QC, SP_QH, SP_QI, SP_VTH, SP_VTI, SP_RHO, SP_EST,
    SP_QSAT, SP_WC, SP_WT, SP_TT, SP_QVT, SP_QCT, SP_QHT, SP_QIT, SP_VCTR1,
    SP_VCTR2, SP_VT3DC, SP_VT3DF, SP_VT3DK, SP_VT3DG, SP_SCR1, SP_PKE,
    SP_THE, SP_THVE, SP_TE, SP_PE, SP_QVENV, SP_VISC, SP_CVI, SP_RADIUS,
    SP_UPE, SP_VPE, SP_VEL_E, SP_VEL_P, SP_RAD_P, SP_VEL_T, SP_RAD_T,
    SP_ZTOP_, SP_FLXW, SP_DN0
};

struct SpScal {
    float wbar, dqsdz, viscosity, tstpf, ztop, area, rsurf, alpha;
    float fmoist, dt, time, tdur, heat_flux_w, wmax, ztopmax;
    int mintime, mdur, maxtime, nm1, kkmax, done, status, kmt;
};

struct SpCol {
    float *s;           // shared arrays
    SpScal *S;          // shared scalars
    int tx, nt;         // thread, threads
    const float *u, *v, *theta, *pi, *qv, *zt, *zm;
    int m1, ncol, col;
    const float *heat_flux;
};

#define A(arr, k) c.s[(arr) * SP_LEV + (k)]
// Levels lo..hi owned by this thread: k with (k - 1) % nt == tx.
#define SP_FOR(k, lo, hi) \
    for (int k = sp_first(c, (lo)); k <= (hi); k += c.nt)
#define SP_SYNC() __syncthreads()

__device__ __forceinline__ int sp_first(const SpCol &c, int lo)
{
    int k = c.tx + 1;
    if (k < lo) k += ((lo - k + c.nt - 1) / c.nt) * c.nt;
    return k;
}

__device__ __forceinline__ float sp_max(float a, float b)
{ return a > b ? a : b; }
__device__ __forceinline__ float sp_min(float a, float b)
{ return a < b ? a : b; }

// set_grid
__device__ __forceinline__ float sp_zt(int k)
{ return k == 1 ? 0.0f : 50.0f + 100.0f * (float)(k - 2); }
__device__ __forceinline__ float sp_zm(int k)
{ return 100.0f * (float)(k - 1); }
__device__ __forceinline__ float sp_dzm(int k)
{
    if (k >= SP_NKP) k = SP_NKP - 1;
    return __fdiv_rn(1.0f, sp_zt(k + 1) - sp_zt(k));
}
__device__ __forceinline__ float sp_dzt(int k)
{
    if (k == 1) {
        const float d2 = __fdiv_rn(1.0f, sp_zm(2) - sp_zm(1));
        const float d3 = __fdiv_rn(1.0f, sp_zm(3) - sp_zm(2));
        return (d2 * d2) / d3;
    }
    return __fdiv_rn(1.0f, sp_zm(k) - sp_zm(k - 1));
}

// The WRF column, one-based like plumerise's arrays.
__device__ __forceinline__ float sp_in(const SpCol &c, const float *a, int k)
{ return a[(size_t)(k - 1) * (size_t)c.ncol + (size_t)c.col]; }

// zzcon, the module's nkp-long copy of the column's w levels: the column's
// levels, then the zeros of the first-call state.
__device__ __forceinline__ float sp_zzcon(const SpCol &c, int k)
{ return (k >= 1 && k <= c.m1) ? sp_in(c, c.zm, k) : 0.0f; }

// ESAT_PR, lines 2436-2463 (kPa).
__device__ float sp_esat(float tem)
{
    const float temc = tem - 273.3f;
    float esatm;
    if (temc > -40.0f) {
        esatm = 6.1121f * gfk_exp(((18.729f - __fdiv_rn(temc, 227.3f)) * temc)
                                  / (temc + 257.87f));
    } else {
        esatm = 6.1115f * gfk_exp((22.542f * temc) / (temc + 273.48f));
    }
    return __fdiv_rn(esatm, 10.0f);
}

// htint, lines 2395-2434 (thread 0): vctra, eleva are WRF-column inputs,
// vctrb a shared array on the plume grid elevb = zt.
__device__ bool sp_htint(SpCol &c, int nzz1, const float *vctra,
                         const float *eleva, int nzz2, int vctrb)
{
    int l = 1;
    for (int k = 1; k <= nzz2; ++k) {
        const float eb = sp_zt(k);
        while (true) {
            const float el = sp_in(c, eleva, l), el1 = sp_in(c, eleva, l + 1);
            if ((eb < sp_in(c, eleva, 1)) || ((eb >= el) && (eb <= el1))) {
                const float wt = (eb - el) / (el1 - el);
                const float va = sp_in(c, vctra, l);
                A(vctrb, k) = va + (sp_in(c, vctra, l + 1) - va) * wt;
                break;
            } else if (eb > sp_in(c, eleva, nzz1)) {
                const float en = sp_in(c, eleva, nzz1);
                const float wt = (eb - en) / (sp_in(c, eleva, nzz1 - 1) - en);
                const float vn = sp_in(c, vctra, nzz1);
                A(vctrb, k) = vn + (sp_in(c, vctra, nzz1 - 1) - vn) * wt;
                break;
            }
            l = l + 1;
            if (l == nzz1) return false;       // WRF: STOP 'htint'
        }
    }
    return true;
}

// get_env_condition(1, m1, kmt, wind_eff = 1), lines 208-288.
__device__ void sp_get_env_condition(SpCol &c)
{
    SpScal &S = *c.S;
    if (c.tx == 0) {
        const float znz = sp_in(c, c.zt, c.m1);
        int k;
        for (k = SP_NKP; k >= 1; --k)
            if (sp_zt(k) < znz) break;
        if (k < 1) {
            S.status = 2;                      // WRF: stop ' envir stop 12'
        } else {
            const int kmt = k < SP_NKP - 1 ? k : SP_NKP - 1;
            S.kmt = kmt;
            const int nk = c.m1;
            if (!sp_htint(c, nk, c.u, c.zt, kmt, SP_UPE)
                || !sp_htint(c, nk, c.v, c.zt, kmt, SP_VPE)
                || !sp_htint(c, nk, c.theta, c.zt, kmt, SP_THE)
                || !sp_htint(c, nk, c.qv, c.zt, kmt, SP_QVENV)) {
                S.status = 1;
            } else {
                for (k = 1; k <= kmt; ++k)
                    A(SP_QVENV, k) = sp_max(A(SP_QVENV, k), 1e-8f);
                A(SP_PKE, 1) = sp_in(c, c.pi, 1);
                for (k = 1; k <= kmt; ++k)
                    A(SP_THVE, k) = A(SP_THE, k)
                                    * (1.0f + 0.61f * A(SP_QVENV, k));
                const float g2 = 9.81f * 2.0f;
                for (k = 2; k <= kmt; ++k)
                    A(SP_PKE, k) = A(SP_PKE, k - 1)
                        - (g2 * (sp_zt(k) - sp_zt(k - 1)))
                          / (A(SP_THVE, k) + A(SP_THVE, k - 1));
            }
        }
    }
    SP_SYNC();
    if (S.status) return;
    const float cp = (7.0f * 287.0f) * 0.5f;      // 7.*r_d/2. (exact)
    const float rcp = __fdiv_rn(287.0f, cp);
    const float cpor = __fdiv_rn(1.0f, rcp);
    SP_FOR(k, 1, S.kmt) {
        const float pke = A(SP_PKE, k);
        A(SP_TE, k) = __fdiv_rn(A(SP_THE, k) * pke, cp);
        A(SP_PE, k) = gfk_pow(__fdiv_rn(pke, cp), cpor) * 100000.0f;
        const float up = A(SP_UPE, k), vp = A(SP_VPE, k);
        A(SP_VEL_E, k) = sqrtf(up * up + vp * vp);
        A(SP_PE, k) = A(SP_PE, k) * 1.e-3f;
    }
    SP_SYNC();
}

// get_fire_properties, lines 415-536 (use_last = 0), thread 0; HEATING is
// formed on the fly by sp_heating.
__device__ void sp_get_fire_properties(SpCol &c, int imm, int iveg,
                                       float burnt_area)
{
    SpScal &S = *c.S;
    S.area = burnt_area;
    S.heat_flux_w = c.heat_flux[(imm - 1) * 4 + (iveg - 1)] * 1000.0f;
    S.mdur = 53;
    S.maxtime = S.mdur + 2;
    S.alpha = 0.05f;
    S.maxtime = S.maxtime * 60;
    S.rsurf = sqrtf(__fdiv_rn(S.area, 3.14159f));
    S.fmoist = __fdiv_rn((float)10, 100.0f);
    S.tdur = (float)S.mdur * 60.0f;
}

__device__ float sp_heating(const SpScal &S, int minute)
{
    const float full = S.heat_flux_w * 0.55f;
    const float hinc = __fdiv_rn(full, 4.0f);
    if (minute == 1) return 0.1f;
    if (minute == 2) return hinc;
    if (minute == 3) return 2.0f * hinc;
    if (minute == 4) return 3.0f * hinc;
    if (minute <= S.mdur) return full;
    return 0.0001f;
}

// ------------------------------------------------------------ microphysics
// One level L with WATERBAL's module scalars wbar, dqsdz and the step dt.

__device__ void sp_evaporate(SpCol &c, int L, float wbar, float dqsdz,
                             float dt)
{
    const float frc = __fdiv_rn(2.5E3f, 1.004f);
    const float src = __fdiv_rn(2834.0f, 1.004f);
    float &qsat = A(SP_QSAT, L), &qv = A(SP_QV, L);
    float &qc = A(SP_QC, L), &qh = A(SP_QH, L);
    float &qi = A(SP_QI, L), &t = A(SP_T, L);
    float sd = qsat - qv;
    if (sd == 0.0f) return;
    const float evrate = fabsf(wbar * dqsdz);
    const float evap = evrate * dt;
    if (sd <= 0.0f) {
        if (evap >= fabsf(sd)) {
            qc = qc - sd;
            qv = qsat;
            t = t - sd * frc;
            return;
        }
        qc = qc + evap;
        qv = qv - evap;
        t = t + evap * frc;
        return;
    }
    if (evap <= qc) {
        if (sd <= evap) {
            qc = qc - sd;
            qv = qsat;
            t = t - sd * frc;
            return;
        }
        sd = sd - evap;
        qv = qv + evap;
        t = t - evap * frc;
        qc = qc - evap;
    } else {
        if (sd <= qc) {
            qv = qsat;
            qc = qc - sd;
            t = t - sd * frc;
            return;
        }
        sd = sd - qc;
        qv = qv + qc;
        t = t - qc * frc;
        qc = 0.0f;
    }
    if (qh > 1.E-10f) {
        const float rho = A(SP_RHO, L);
        const float quant = ((qsat - qc) - qv) * rho;
        const float evhdt = (((dt * 5.44E-4f) * quant)
                             * gfk_pow(qh * rho, .65f)) / rho;
        if (evhdt <= qh) {
            if (sd <= evhdt) {
                qh = qh - sd;
                qv = qsat;
                t = t - sd * frc;
                return;
            }
            sd = sd - evhdt;
            qv = qv + evhdt;
            t = t - evhdt * frc;
            qh = qh - evhdt;
        } else {
            if (sd <= qh) {
                qv = qsat;
                qh = qh - sd;
                t = t - sd * frc;
                return;
            }
            sd = sd - qh;
            qv = qv + qh;
            t = t - qh * frc;
            qh = 0.0f;
        }
    }
    if (qi <= 1.E-10f) return;
    const float dividend =
        ((gfk_pow(__fdiv_rn(1.E6f, A(SP_RHO, L)), 0.475f)
          * (sd / qsat - 1.0f)) * gfk_pow(qi, 0.525f)) * 1.13f;
    const float divisor = 7.E5f + __fdiv_rn(4.1E6f, 10.0f * A(SP_EST, L));
    const float devidt = -((A(SP_CVI, L) * dividend) / divisor);
    const float evidt = devidt * dt;
    if (evidt <= qi) {
        if (sd <= evidt) {
            qi = qi - sd;
            qv = qsat;
            t = t - sd * src;
            return;
        }
        sd = sd - evidt;
        qv = qv + evidt;
        t = t - evidt * src;
        qi = qi - evidt;
    } else {
        if (sd <= qi) {
            qv = qsat;
            qi = qi - sd;
            t = t - sd * src;
            return;
        }
        sd = sd - qi;
        qv = qv + qi;
        t = t - qi * src;
        qi = 0.0f;
    }
}

__device__ void sp_sublimate(SpCol &c, int L, float dt)
{
    const float src = __fdiv_rn(2834.0f, 1.004f);
    float &qv = A(SP_QV, L), &qi = A(SP_QI, L), &t = A(SP_T, L);
    const float qsat = A(SP_QSAT, L);
    if (t > 269.3f) return;
    if (qv <= qsat) return;
    const float dividend =
        ((gfk_pow(__fdiv_rn(1.E6f, A(SP_RHO, L)), 0.475f)
          * (qv / qsat - 1.0f)) * gfk_pow(qi, 0.525f)) * 1.13f;
    const float divisor = 7.E5f + __fdiv_rn(4.1E6f, 10.0f * A(SP_EST, L));
    const float dtsubh = fabsf(dividend / divisor);
    const float subl = dtsubh * dt;
    if (subl < qv) {
        qv = qv - subl;
        qi = qi + subl;
        t = t + subl * src;
        return;
    }
    qi = qv;                 // WRF's own: the ice becomes the vapour
    t = t + qv * src;
    qv = 0.0f;
}

__device__ void sp_glaciate(SpCol &c, int L, float dt)
{
    const float frc = __fdiv_rn(334.0f, 1.004f);
    float &qh = A(SP_QH, L), &qi = A(SP_QI, L), &t = A(SP_T, L);
    if (qh <= 0.0f) return;
    if (A(SP_QV, L) < A(SP_QSAT, L)) return;
    if (t > 269.3f) return;
    const float dfrzh = (dt * 0.025f) * qh;
    if (dfrzh < qh) {
        qi = qi + dfrzh;
        qh = qh - dfrzh;
        t = t + frc * dfrzh;
        return;
    }
    qi = qi + qh;
    t = t + frc * qh;
    qh = 0.0f;
}

__device__ void sp_melt(SpCol &c, int L, float dt)
{
    float &qh = A(SP_QH, L), &qi = A(SP_QI, L), &t = A(SP_T, L);
    if (qi <= 0.0f) return;
    if (t < 273.0f) return;
    const float rho = A(SP_RHO, L);
    // F0**(-0.42): a constant expression gfortran folds; its word is
    // powf's (checked by the parity test's literal pass).
    const float f0pow = gfk_pow(0.75f, -0.42f);
    const float dtmelt = ((((dt * __fdiv_rn(2.27f, rho)) * A(SP_CVI, L))
                           * (t - 273.0f))
                          * gfk_pow((rho * qi) * 1.E-6f, 0.525f)) * f0pow;
    if (dtmelt < qi) {
        qh = qh + dtmelt;
        qi = qi - dtmelt;
        t = t - 332.27f * dtmelt;
        return;
    }
    qh = qh + qi;
    t = t - 332.27f * qi;
    qi = 0.0f;
}

__device__ void sp_convert(SpCol &c, int L, float dt)
{
    float &qc = A(SP_QC, L), &qh = A(SP_QH, L);
    if (A(SP_T, L) <= 269.3f) return;
    if (qc == 0.0f) return;
    float accrete = 0.0f;
    const float rho = A(SP_RHO, L);
    const float q = rho * qc;
    const float h = rho * qh;
    if (qh > 0.0f) accrete = (0.0052f * q) * gfk_pow(h, .875f);
    const float bdisp = 0.146f;
    const float con = (((q * q) * q) * bdisp)
                      / (60.0f * (((5.0f * q) * bdisp) + 0.0366f * 100000.0f));
    const float total = ((con + accrete) * dt) / rho;
    if (total < qc) {
        qc = qc - total;
        qh = qh + total;
        return;
    }
    qh = qh + qc;
    qc = 0.0f;
}

__device__ void sp_waterbal(SpCol &c, int L, float wbar, float dqsdz,
                            float dt)
{
    if (A(SP_QC, L) <= 1.0E-10f) A(SP_QC, L) = 0.0f;
    if (A(SP_QH, L) <= 1.0E-10f) A(SP_QH, L) = 0.0f;
    if (A(SP_QI, L) <= 1.0E-10f) A(SP_QI, L) = 0.0f;
    sp_evaporate(c, L, wbar, dqsdz, dt);
    sp_sublimate(c, L, dt);
    sp_glaciate(c, L, dt);
    sp_melt(c, L, dt);
    sp_convert(c, L, dt);
}

// --------------------------------------------------------- boundary, start

// LBOUND (thread 0), lines 869-965, with BURN.
__device__ void sp_lbound(SpCol &c)
{
    SpScal &S = *c.S;
    A(SP_QH, 1) = A(SP_QH, 2);
    A(SP_QI, 1) = A(SP_QI, 2);
    A(SP_QC, 1) = 0.0f;
    float eflux, water;
    if (S.time > S.tdur) {
        eflux = 0.000001f;
        water = 0.0f;
    } else {
        eflux = sp_heating(S, S.mintime);
        water = __fdiv_rn((eflux * __fdiv_rn(S.dt, 19.3e6f))
                          * (0.5f + S.fmoist), 0.55f);
        water = water * 1000.0f;
    }
    const float pres = A(SP_PE, 1) * 1000.0f;
    const float c1 = 5.0f / (6.0f * S.alpha);
    const float c2 = 0.9f * S.alpha;
    float f = eflux / ((pres * 1004.6f) * 3.14159f);
    f = ((9.80796f * 287.04f) * f) * S.area;
    const float zv = c1 * S.rsurf;
    const float e1 = __fdiv_rn(1.0f, 3.0f), e2 = __fdiv_rn(5.0f, 3.0f);
    const float c2f_e1 = gfk_pow(c2 * f, e1);
    A(SP_W, 1) = (c1 * c2f_e1) / gfk_pow(zv, e1);
    const float denscor = ((__fdiv_rn(c1 * f, 9.80796f)) / c2f_e1)
                          / gfk_pow(zv, e2);
    A(SP_T, 1) = A(SP_TE, 1) / (1.0f - denscor);
    A(SP_WC, 1) = A(SP_W, 1);
    A(SP_VEL_P, 1) = 0.0f;
    A(SP_RAD_P, 1) = S.rsurf;
    A(SP_VTH, 1) = -4.0f;
    A(SP_VTI, 1) = -3.0f;
    A(SP_VISC, 1) = S.viscosity;
    A(SP_RHO, 1) = (3483.8f * A(SP_PE, 1)) / A(SP_T, 1);
    const float xwater = water / ((A(SP_W, 1) * S.dt) * A(SP_RHO, 1));
    A(SP_QV, 1) = xwater + A(SP_QVENV, 1);
    const float es = sp_esat(A(SP_T, 1));
    A(SP_EST, 1) = es;
    A(SP_QSAT, 1) = (0.622f * es) / (A(SP_PE, 1) - es);
    if (A(SP_QV, 1) > A(SP_QSAT, 1)) {
        A(SP_QC, 1) = (A(SP_QV, 1) - A(SP_QSAT, 1)) + A(SP_QC, 1);
        A(SP_QV, 1) = A(SP_QSAT, 1);
    }
    sp_waterbal(c, 1, S.wbar, S.dqsdz, S.dt);
}

// INITIAL, lines 968-1030.
__device__ void sp_initial(SpCol &c)
{
    SpScal &S = *c.S;
    const int n = S.kmt;
    SP_FOR(k, 1, n) {
        A(SP_W, k) = 0.0f;
        A(SP_T, k) = A(SP_TE, k);
        A(SP_WC, k) = 0.0f;
        A(SP_WT, k) = 0.0f;
        A(SP_QV, k) = A(SP_QVENV, k);
        A(SP_VTH, k) = 0.0f;
        A(SP_VTI, k) = 0.0f;
        A(SP_QH, k) = 0.0f;
        A(SP_QI, k) = 0.0f;
        A(SP_QC, k) = 0.0f;
        const float es = sp_esat(A(SP_T, k));
        A(SP_EST, k) = es;
        A(SP_QSAT, k) = (.622f * es) / (A(SP_PE, k) - es);
        A(SP_RHO, k) = (3483.8f * A(SP_PE, k)) / A(SP_T, k);
        A(SP_VEL_P, k) = 0.0f;
        A(SP_RAD_P, k) = 0.0f;
    }
    SP_SYNC();
    if (c.tx == 0) {
        const float grow = 1.2f * S.alpha;
        A(SP_RADIUS, 1) = S.rsurf;
        for (int k = 2; k <= n; ++k)
            A(SP_RADIUS, k) = A(SP_RADIUS, k - 1)
                              + grow * (sp_zt(k) - sp_zt(k - 1));
        A(SP_RADIUS, 1) = S.rsurf;
        A(SP_RAD_P, 1) = S.rsurf;
        for (int k = 2; k <= n; ++k) {
            A(SP_RADIUS, k) = A(SP_RADIUS, k - 1)
                              + grow * (sp_zt(k) - sp_zt(k - 1));
            A(SP_RAD_P, k) = A(SP_RADIUS, k);
        }
        A(SP_VISC, 1) = S.viscosity;
        const float step = __fdiv_rn(1.0f * S.viscosity, (float)SP_NKP);
        for (int k = 2; k <= n; ++k)
            A(SP_VISC, k) = sp_max(1.e-3f, A(SP_VISC, k - 1) - step);
        sp_lbound(c);
    }
    SP_SYNC();
}

// ---------------------------------------------------------------- dynamics

// fa_zc_plumerise then advtndc_plumerise for one scalar scp with tendency
// sct; SP_SCR1 is the advected copy.
__device__ void sp_advect_scalar(SpCol &c, int m1, int scp, int sct,
                                 float dtli)
{
    SP_FOR(k, 1, m1) A(SP_SCR1, k) = A(scp, k);
    SP_SYNC();
    SP_FOR(k, 1, m1 - 1) {
        float g = A(SP_VT3DC, k)
            * ((A(SP_VCTR1, k) * A(SP_SCR1, k)
                + A(SP_VCTR2, k) * A(SP_SCR1, k + 1))
               + A(SP_VT3DF, k) * (A(SP_SCR1, k) - A(SP_SCR1, k + 1)));
        const float vc = A(SP_VT3DC, k);
        if (vc > 0.0f) {
            if (g * A(SP_VT3DK, k) > .5f * A(SP_SCR1, k))
                g = vc * A(SP_SCR1, k);
        } else if (vc < 0.0f) {
            if (-g * A(SP_VT3DK, k + 1) > .5f * A(SP_SCR1, k + 1))
                g = vc * A(SP_SCR1, k + 1);
        }
        A(SP_VT3DG, k) = g;
    }
    SP_SYNC();
    SP_FOR(k, 2, m1 - 1) {
        A(SP_SCR1, k) = A(SP_SCR1, k) + A(SP_VT3DK, k)
            * ((A(SP_VT3DG, k - 1) - A(SP_VT3DG, k))
               + A(scp, k) * (A(SP_VT3DC, k) - A(SP_VT3DC, k - 1)));
        A(sct, k) = A(sct, k) + (A(SP_SCR1, k) - A(scp, k)) * dtli;
    }
    SP_SYNC();
}

// One makeplume time step, lines 651-805; returns through S.done.
__device__ void sp_step(SpCol &c)
{
    SpScal &S = *c.S;
    const int deltak = 20;
    const int kmt = S.kmt;
    if (c.tx == 0) {
        S.nm1 = kmt < S.kkmax + deltak ? kmt : S.kkmax + deltak;
        S.dt = sp_min(5.0f, (sp_zm(2) - sp_zm(1)) / (S.tstpf * S.wmax));
        S.time = S.time + S.dt;
        S.mintime = 1 + (int)S.time / 60;
        S.wmax = 1.0f;
    }
    SP_SYNC();
    const int nm1 = S.nm1;
    const float dt = S.dt;
    // tend0_plumerise; lbound on thread 0, which owns level 1.
    SP_FOR(k, 1, nm1) {
        A(SP_WT, k) = 0.0f;
        A(SP_TT, k) = 0.0f;
        A(SP_QVT, k) = 0.0f;
        A(SP_QCT, k) = 0.0f;
        A(SP_QHT, k) = 0.0f;
        A(SP_QIT, k) = 0.0f;
        A(SP_VEL_T, k) = 0.0f;
        A(SP_RAD_T, k) = 0.0f;
    }
    if (c.tx == 0) sp_lbound(c);
    SP_SYNC();
    // vel_advectc_plumerise(nm1, wc, wt, rho, dzm)
    SP_FOR(k, 1, nm1) A(SP_DN0, k) = A(SP_RHO, k) * 1.e-3f;
    SP_SYNC();
    SP_FOR(k, 1, nm1 - 1) {
        if (k == 1)
            A(SP_FLXW, 1) = A(SP_WC, 1) * A(SP_DN0, 1);
        else
            A(SP_FLXW, k) = (A(SP_WC, k) * .5f)
                            * (A(SP_DN0, k) + A(SP_DN0, k + 1));
    }
    SP_SYNC();
    SP_FOR(k, 2, nm1 - 2) {
        const float fk = A(SP_FLXW, k), fm = A(SP_FLXW, k - 1),
                    fp = A(SP_FLXW, k + 1);
        const float wk = A(SP_WC, k), wm = A(SP_WC, k - 1),
                    wp = A(SP_WC, k + 1);
        const float a = (fk + fm) * (wk + wm);
        const float b = (fk + fp) * (wk + wp);
        const float cc = ((fp - fm) * 2.0f) * wk;
        A(SP_WT, k) = A(SP_WT, k)
            + ((.5f * sp_dzm(k)) / (A(SP_DN0, k) + A(SP_DN0, k + 1)))
              * ((a - b) + cc);
    }
    // scl_advectc_plumerise('SC', nm1)
    const float dtlto2 = .5f * dt;
    SP_FOR(k, 1, nm1) {
        if (k == 1) {
            A(SP_VT3DC, 1) = (((A(SP_W, 1) + A(SP_WC, 1)) * dtlto2)
                              * A(SP_RHO, 1)) * 1.e-3f;
            A(SP_VT3DF, 1) = ((.5f * (A(SP_W, 1) + A(SP_WC, 1))) * dtlto2)
                             * sp_dzm(1);
        } else {
            const float s = (A(SP_W, k) + A(SP_WC, k)) * dtlto2;
            A(SP_VT3DC, k) = ((s * .5f) * (A(SP_RHO, k) + A(SP_RHO, k + 1)))
                             * 1.e-3f;
            A(SP_VT3DF, k) = (s * .5f) * sp_dzm(k);
        }
        A(SP_VCTR1, k) = (sp_zt(k + 1) - sp_zm(k)) * sp_dzm(k);
        A(SP_VCTR2, k) = (sp_zm(k) - sp_zt(k)) * sp_dzm(k);
        A(SP_VT3DK, k) = sp_dzt(k) / (A(SP_RHO, k) * 1.e-3f);
    }
    SP_SYNC();
    const float dtli = 1.0f / dt;
    sp_advect_scalar(c, nm1, SP_T, SP_TT, dtli);
    sp_advect_scalar(c, nm1, SP_QV, SP_QVT, dtli);
    sp_advect_scalar(c, nm1, SP_QC, SP_QCT, dtli);
    sp_advect_scalar(c, nm1, SP_QI, SP_QIT, dtli);
    sp_advect_scalar(c, nm1, SP_QH, SP_QHT, dtli);
    sp_advect_scalar(c, nm1, SP_VEL_P, SP_VEL_T, dtli);
    sp_advect_scalar(c, nm1, SP_RAD_P, SP_RAD_T, dtli);
    // scl_misc, scl_dyn_entrain, damp_grav_wave(1): level by level.  WBAR
    // set here is overwritten by the microphysics before anything reads it.
    {
        const float c2pi = __fdiv_rn(2.0f, 3.1416f);
        const int kf = nm1 - deltak;
        const float zmkf = sp_zm(kf);
        const float ztopf = sp_zm(nm1);
        const float distim = sp_min(3.0f * dt, 60.0f);
        const float c1 = 1.0f / (distim * (ztopf - zmkf));
        SP_FOR(k, 2, nm1 - 1) {
            const float wbar = .5f * (A(SP_W, k) + A(SP_W, k - 1));
            const float adiabat = -__fdiv_rn(wbar * 9.81f, 1004.0f);
            float dmdtm = ((2.0f * S.alpha) * fabsf(wbar)) / A(SP_RADIUS, k);
            A(SP_TT, k) = (A(SP_TT, k) + adiabat)
                          - dmdtm * (A(SP_T, k) - A(SP_TE, k));
            A(SP_QVT, k) = A(SP_QVT, k)
                           - dmdtm * (A(SP_QV, k) - A(SP_QVENV, k));
            A(SP_QCT, k) = A(SP_QCT, k) - dmdtm * A(SP_QC, k);
            A(SP_QHT, k) = A(SP_QHT, k) - dmdtm * A(SP_QH, k);
            A(SP_QIT, k) = A(SP_QIT, k) - dmdtm * A(SP_QI, k);
            A(SP_VEL_T, k) = A(SP_VEL_T, k)
                             - dmdtm * (A(SP_VEL_P, k) - A(SP_VEL_E, k));
            A(SP_RAD_T, k) = A(SP_RAD_T, k)
                             + ((.5f * dmdtm) * 1.2f) * A(SP_RADIUS, k);
            const float dv = fabsf(A(SP_VEL_E, k) - A(SP_VEL_P, k));
            A(SP_RAD_T, k) = A(SP_RAD_T, k) + __fdiv_rn(dv, 3.1416f);
            dmdtm = (c2pi * dv) / A(SP_RADIUS, k);
            A(SP_VEL_T, k) = A(SP_VEL_T, k)
                             - dmdtm * (A(SP_VEL_P, k) - A(SP_VEL_E, k));
            A(SP_TT, k) = A(SP_TT, k) - dmdtm * (A(SP_T, k) - A(SP_TE, k));
            A(SP_QVT, k) = A(SP_QVT, k)
                           - dmdtm * (A(SP_QV, k) - A(SP_QVENV, k));
            A(SP_QCT, k) = A(SP_QCT, k) - dmdtm * A(SP_QC, k);
            A(SP_QHT, k) = A(SP_QHT, k) - dmdtm * A(SP_QH, k);
            A(SP_QIT, k) = A(SP_QIT, k) - dmdtm * A(SP_QI, k);
        }
        // friction(1) runs over k = nm1 .. 2, so level nm1 too.
        SP_FOR(k, 2, nm1) {
            const float z = sp_zt(k);
            if (z > zmkf)
                A(SP_TT, k) = A(SP_TT, k) + (c1 * (z - zmkf))
                              * (A(SP_TE, k) - A(SP_T, k));
        }
    }
    SP_SYNC();
    // Microphysics, three substeps of dt/3.
    const float dtm = __fdiv_rn(dt, (float)3);
    for (int sub = 0; sub < 3; ++sub) {
        // fallpart(nm1): VTH(k+1), VTI(k+1) are read before k+1 is set.
        float vth_up[4], vti_up[4], qsat_up[4];
        {
            int i = 0;
            SP_FOR(k, 2, nm1 - 1) {
                vth_up[i] = A(SP_VTH, k + 1);
                vti_up[i] = A(SP_VTI, k + 1);
                ++i;
            }
        }
        SP_SYNC();
        SP_FOR(k, 2, nm1 - 1) {
            A(SP_VTH, k) = -4.0f;
            A(SP_VTI, k) = -3.0f;
        }
        SP_SYNC();
        {
            int i = 0;
            SP_FOR(k, 2, nm1 - 1) {
                const float vhrel = A(SP_W, k) + A(SP_VTH, k);
                const float virel = A(SP_W, k) + A(SP_VTI, k);
                A(SP_CVI, k) = 1.6f + __fdiv_rn(0.57E-3f
                                                * gfk_pow(fabsf(virel), 1.5f),
                                                0.75f);
                const float rk = A(SP_RHO, k), rm = A(SP_RHO, k - 1),
                            rp = A(SP_RHO, k + 1);
                float dfhz, dfiz;
                if (vhrel >= 0.0f)
                    dfhz = (A(SP_QH, k) * (rk * A(SP_VTH, k)
                                           - rm * A(SP_VTH, k - 1))) / rm;
                else
                    dfhz = (A(SP_QH, k) * (rp * vth_up[i]
                                           - rk * A(SP_VTH, k))) / rk;
                if (virel >= 0.0f)
                    dfiz = (A(SP_QI, k) * (rk * A(SP_VTI, k)
                                           - rm * A(SP_VTI, k - 1))) / rm;
                else
                    dfiz = (A(SP_QI, k) * (rp * vti_up[i]
                                           - rk * A(SP_VTI, k))) / rk;
                const float dz1 = sp_zm(k) - sp_zm(k - 1);
                A(SP_QHT, k) = A(SP_QHT, k) - dfhz / dz1;
                A(SP_QIT, k) = A(SP_QIT, k) - dfiz / dz1;
                qsat_up[i] = A(SP_QSAT, k + 1);    // QSAT(L+1) as it stands
                ++i;
            }
        }
        SP_SYNC();
        SP_FOR(k, 2, nm1 - 1) {
            const float es = sp_esat(A(SP_T, k));
            A(SP_QSAT, k) = (0.622f * es) / (A(SP_PE, k) - es);
            A(SP_EST, k) = es;
            A(SP_RHO, k) = (3483.8f * A(SP_PE, k)) / A(SP_T, k);
        }
        SP_SYNC();
        {
            int i = 0;
            SP_FOR(L, 2, nm1 - 1) {
                const float wbar = .5f * (A(SP_W, L) + A(SP_W, L - 1));
                const float dqsdz = (qsat_up[i] - A(SP_QSAT, L - 1))
                                    / (sp_zt(L + 1) - sp_zt(L - 1));
                sp_waterbal(c, L, wbar, dqsdz, dtm);
                if (L == nm1 - 1) {
                    S.wbar = wbar;
                    S.dqsdz = dqsdz;
                }
                ++i;
            }
        }
        SP_SYNC();
    }
    // visc_W(nm1, deltak, kmt)
    {
        const int m2 = nm1 < kmt ? nm1 : kmt;
        SP_FOR(k, 2, m2 - 1) {
            const float dz1t = .5f * (sp_zt(k + 1) - sp_zt(k - 1));
            const float dz2t = A(SP_VISC, k) / (dz1t * dz1t);
            const float dz1m = .5f * (sp_zm(k + 1) - sp_zm(k - 1));
            const float dz2m = A(SP_VISC, k) / (dz1m * dz1m);
#define SP_D2(arr, s) (((A(arr, k + 1) - 2.0f * A(arr, k)) + A(arr, k - 1)) \
                       * (s))
            A(SP_WT, k) = A(SP_WT, k) + SP_D2(SP_W, dz2m);
            A(SP_TT, k) = A(SP_TT, k) + SP_D2(SP_T, dz2t);
            A(SP_QVT, k) = A(SP_QVT, k) + SP_D2(SP_QV, dz2t);
            A(SP_QCT, k) = A(SP_QCT, k) + SP_D2(SP_QC, dz2t);
            A(SP_QHT, k) = A(SP_QHT, k) + SP_D2(SP_QH, dz2t);
            A(SP_QIT, k) = A(SP_QIT, k) + SP_D2(SP_QI, dz2t);
            A(SP_VEL_T, k) = A(SP_VEL_T, k) + SP_D2(SP_VEL_P, dz2t);
            A(SP_RAD_T, k) = A(SP_RAD_T, k) + SP_D2(SP_RAD_P, dz2t);
#undef SP_D2
        }
    }
    SP_SYNC();
    // update_plumerise(nm1, 'S'), hadvance_plumerise(1), the buoyancy
    // field: level by level.
    const float eps = S.mintime == 1 ? 0.5f : .2f;
    const float umgamai = 1.0f / (1.0f + 0.5f);
    SP_FOR(k, 1, nm1) {
        if (k >= 2 && k <= nm1 - 1) {
            A(SP_T, k) = A(SP_T, k) + A(SP_TT, k) * dt;
            A(SP_QV, k) = A(SP_QV, k) + A(SP_QVT, k) * dt;
            A(SP_QC, k) = A(SP_QC, k) + A(SP_QCT, k) * dt;
            A(SP_QH, k) = A(SP_QH, k) + A(SP_QHT, k) * dt;
            A(SP_QI, k) = A(SP_QI, k) + A(SP_QIT, k) * dt;
            A(SP_QV, k) = sp_max(0.0f, A(SP_QV, k));
            A(SP_QC, k) = sp_max(0.0f, A(SP_QC, k));
            A(SP_QH, k) = sp_max(0.0f, A(SP_QH, k));
            A(SP_QI, k) = sp_max(0.0f, A(SP_QI, k));
            A(SP_VEL_P, k) = A(SP_VEL_P, k) + A(SP_VEL_T, k) * dt;
            A(SP_RAD_P, k) = A(SP_RAD_P, k) + A(SP_RAD_T, k) * dt;
        }
        A(SP_WC, k) = A(SP_WC, k) + eps * (A(SP_W, k) - 2.0f * A(SP_WC, k));
        if (k >= 2 && k <= nm1 - 1) {
            const float qv = A(SP_QV, k), qve = A(SP_QVENV, k);
            const float tv = (A(SP_T, k) * (1.0f + __fdiv_rn(qv, 0.622f)))
                             / (1.0f + qv);
            const float tve = (A(SP_TE, k) * (1.0f + __fdiv_rn(qve, 0.622f)))
                              / (1.0f + qve);
            const float qw = (A(SP_QH, k) + A(SP_QI, k)) + A(SP_QC, k);
            A(SP_SCR1, k) = (9.8f * umgamai) * ((tv - tve) / tve - qw);
        }
    }
    SP_SYNC();
    // The buoyancy and entrainment accelerations, update_plumerise('W'),
    // hadvance_plumerise(2), the misc loop and damp_grav_wave(2).
    {
        const float c2pi_half = __fdiv_rn(2.0f, 3.1416f) * 0.5f;
        const int kf = nm1 - deltak;
        const float zmkf = sp_zm(kf);
        const float ztopf = sp_zm(nm1);
        const float distim = sp_min(3.0f * dt, 60.0f);
        const float c1 = 1.0f / (distim * (ztopf - zmkf));
        const float c2 = dt * c1;
        SP_FOR(k, 1, nm1) {
            if (k >= 2 && k <= nm1 - 2)
                A(SP_WT, k) = A(SP_WT, k)
                              + .5f * (A(SP_SCR1, k) + A(SP_SCR1, k + 1));
            if (k >= 2 && k <= nm1 - 1) {
                const float wbar = A(SP_W, k);
                const float rbar = .5f * (A(SP_RADIUS, k)
                                          + A(SP_RADIUS, k - 1));
                const float dmdtm =
                    (((umgamai * 2.0f) * S.alpha) * fabsf(wbar)) / rbar;
                A(SP_WT, k) = A(SP_WT, k) - dmdtm * fabsf(wbar);
                const float dv = ((A(SP_VEL_P, k) - A(SP_VEL_E, k))
                                  + A(SP_VEL_P, k - 1)) - A(SP_VEL_E, k - 1);
                const float dyn = (c2pi_half * fabsf(dv)) / rbar;
                A(SP_WT, k) = A(SP_WT, k) - dyn * fabsf(wbar);
                A(SP_W, k) = A(SP_W, k) + A(SP_WT, k) * dt;
            }
            const float af = A(SP_W, k);
            A(SP_W, k) = A(SP_WC, k) + eps * af;
            A(SP_WC, k) = af;
            if (k >= 2) {
                const float es = sp_esat(A(SP_T, k));
                A(SP_QSAT, k) = (0.622f * es) / (A(SP_PE, k) - es);
                A(SP_EST, k) = es;
                A(SP_RHO, k) = (3483.8f * A(SP_PE, k)) / A(SP_T, k);
                const float z = sp_zt(k);
                if (z > zmkf)
                    A(SP_W, k) = A(SP_W, k) + (c2 * (z - zmkf))
                                 * (0.0f - A(SP_W, k));
            }
        }
    }
    SP_SYNC();
    // The entrainment above read RADIUS(k-1); it is updated only now.
    SP_FOR(k, 2, nm1) A(SP_RADIUS, k) = A(SP_RAD_P, k);
    if (c.tx == 0) {
        float wmax = S.wmax;
        for (int k = 2; k <= nm1; ++k)
            if (fabsf(A(SP_WC, k)) > wmax) wmax = fabsf(A(SP_WC, k));
        S.wmax = wmax;
        int kk = 1;
        while (A(SP_W, kk) > 1.0f) {
            kk = kk + 1;
            S.ztop = sp_zm(kk);
        }
        A(SP_ZTOP_, S.mintime) = S.ztop;
        S.ztopmax = sp_max(S.ztop, S.ztopmax);
        S.kkmax = kk > S.kkmax ? kk : S.kkmax;
        if (S.mintime > 10) {
            if (fabsf(A(SP_ZTOP_, S.mintime) - A(SP_ZTOP_, S.mintime - 10))
                < 100.0f)
                S.done = 1;
        }
        if (!(S.time <= (float)S.maxtime)) S.done = 1;
    }
    SP_SYNC();
}

// MAKEPLUME, lines 539-832; leaves ztopmax in S.ztopmax.
__device__ void sp_makeplume(SpCol &c)
{
    SpScal &S = *c.S;
    if (c.tx == 0) {
        S.tstpf = 2.0f;
        S.viscosity = 500.0f;
        S.mintime = 1;
        S.ztopmax = 0.0f;
        S.ztop = 0.0f;
        S.time = 0.0f;
        S.dt = 1.0f;
        S.wmax = 1.0f;
        S.kkmax = 10;
        S.done = 0;
    }
    SP_SYNC();
    sp_initial(c);
    if (c.tx == 0 && !(S.time <= (float)S.maxtime)) S.done = 1;
    SP_SYNC();
    while (!S.done) sp_step(c);
}

// One burning column per block: plumerise, lines 10-205, with
// set_flam_vert's k1/k2 search (lines 345-361).  The optional trace
// outputs hold the plume state after every makeplume (slot imm + 2 * biome
// - 3, the order of tools/smoke_wrf471_oracle/run_plumerise.F90).
extern "C" __global__ void smoke_plumerise_columns(
    const int *__restrict__ index, int m1, int ncol,
    const float *u, const float *v, const float *theta, const float *pi,
    const float *qv, const float *zt, const float *zm,
    const float *__restrict__ firesize, const float *__restrict__ mean_fct,
    const float *__restrict__ eburn_in, const float *__restrict__ heat_flux,
    float *__restrict__ eburn_out, float *__restrict__ ztopmax_out,
    int *__restrict__ k1_out, int *__restrict__ k2_out,
    int *__restrict__ status, float *__restrict__ trace_state,
    float *__restrict__ trace_scal, int *__restrict__ trace_min)
{
    extern __shared__ float sp_shared[];
    __shared__ SpScal S;
    SpCol c;
    c.s = sp_shared;
    c.S = &S;
    c.tx = threadIdx.x;
    c.nt = blockDim.x;
    c.u = u; c.v = v; c.theta = theta; c.pi = pi; c.qv = qv;
    c.zt = zt; c.zm = zm;
    c.m1 = m1;
    c.ncol = ncol;
    c.col = index[blockIdx.x];
    c.heat_flux = heat_flux;
    const size_t col = (size_t)c.col, nc = (size_t)ncol;

    // zero_plumegen_coms: the first-call state.
    for (int i = c.tx; i < SP_NARR * SP_LEV; i += c.nt) sp_shared[i] = 0.0f;
    if (c.tx == 0) {
        S.wbar = S.dqsdz = S.viscosity = S.tstpf = S.ztop = 0.0f;
        S.area = S.rsurf = S.alpha = S.fmoist = S.dt = S.time = 0.0f;
        S.tdur = S.heat_flux_w = S.wmax = S.ztopmax = 0.0f;
        S.mintime = S.mdur = S.maxtime = S.nm1 = S.kkmax = 0;
        S.done = S.status = S.kmt = 0;
    }
    SP_SYNC();
    sp_get_env_condition(c);
    if (S.status) {
        if (c.tx == 0) status[col] = S.status;
        return;
    }
    const float ein = eburn_in[col];
    float ztopmax[3] = {0.0f, 0.0f, 0.0f};
    for (int iveg = 1; iveg <= 4; ++iveg) {
        const float fct = mean_fct[(size_t)(iveg - 1) * nc + col];
        if (fct < 1.e-6f) continue;
        const float burnt_area = firesize[(size_t)(iveg - 1) * nc + col];
        for (int imm = 1; imm <= 2; ++imm) {
            if (c.tx == 0) sp_get_fire_properties(c, imm, iveg, burnt_area);
            SP_SYNC();
            if (iveg == 4 && imm == 2) {
                ztopmax[2] = ztopmax[1];
                ztopmax[1] = sp_zzcon(c, 1);
                continue;
            }
            sp_makeplume(c);
            ztopmax[imm] = S.ztopmax;
            if (trace_state) {
                const size_t slot = (size_t)(imm + 2 * (iveg - 1) - 1);
                const int fields[SP_NSTATE] = {
                    SP_W, SP_T, SP_QV, SP_QC, SP_QH, SP_QI, SP_RHO, SP_RADIUS,
                    SP_VEL_P, SP_RAD_P, SP_WC, SP_QSAT};
                for (int j = 0; j < SP_NSTATE; ++j)
                    for (int k = 1 + c.tx; k <= SP_NKP; k += c.nt)
                        trace_state[((slot * SP_NSTATE + j) * SP_NKP + (k - 1))
                                    * nc + col] = A(fields[j], k);
                if (c.tx == 0) {
                    trace_scal[(slot * 4 + 0) * nc + col] = S.time;
                    trace_scal[(slot * 4 + 1) * nc + col] = S.dt;
                    trace_scal[(slot * 4 + 2) * nc + col] = S.wbar;
                    trace_scal[(slot * 4 + 3) * nc + col] = S.dqsdz;
                    trace_min[slot * nc + col] = S.mintime;
                }
            }
            SP_SYNC();
        }
        if (c.tx == 0) {
            int klim[3];
            for (int imm = 1; imm <= 2; ++imm) {
                int k;
                for (k = 1; k <= SP_NKP - 1; ++k)
                    if (sp_zzcon(c, k) > ztopmax[imm]) break;
                klim[imm] = k;
            }
            const int k1 = 3 > klim[1] ? 3 : klim[1];
            int k2 = 3 > klim[2] ? 3 : klim[2];
            if (k2 < k1) k2 = k1;
            const float dz_flam = sp_zzcon(c, k2 + 1) - sp_zzcon(c, k1);
            const float dzi = 1.0f / dz_flam;
            const float add = (fct * ein) * dzi;
            for (int k = k1; k <= k2 && k <= m1; ++k)
                eburn_out[(size_t)(k - 1) * nc + col] =
                    eburn_out[(size_t)(k - 1) * nc + col] + add;
            // k2 >= m1: dz_flam reads zzcon(m1+1), past the column, and
            // k2 > m1 writes past eburn_out; refused, as the reference does.
            if (k2 >= m1) status[col] = 3;
            ztopmax_out[((size_t)(iveg - 1) * 2 + 0) * nc + col] = ztopmax[1];
            ztopmax_out[((size_t)(iveg - 1) * 2 + 1) * nc + col] = ztopmax[2];
            k1_out[(size_t)(iveg - 1) * nc + col] = k1;
            k2_out[(size_t)(iveg - 1) * nc + col] = k2;
        }
    }
}
