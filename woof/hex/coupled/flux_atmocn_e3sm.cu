// E3SM v3.2.0's atmosphere-ocean flux on the coupler's flux grid:
// seq_flux_atmocn_mct (driver-mct/main/seq_flux_mct.F90:1333-1727) over
// shr_flux_atmOcn's E3SMv1 scheme (share/util/shr_flux_mod.F90:144-703,
// ocn_surface_flux_scheme = 0) and shr_flux_update_stress (2498-2554), with
// seq_flux_init_mct's mask (217-500). real(r8); every operation an explicit
// round-to-nearest intrinsic, in the Fortran's order (gfortran -O0
// -ffp-contract=off).
//
// libm: exp and log are glibc's (glibc_exp, glibc_log), x**0.5 is pow
// (glibc_pow), x**4 libgcc's __powidf2, (x*x)*(x*x) (the call gfortran -O0
// makes). psimhu calls glibc's binary64 atan on every iteration, which is not
// correctly rounded and is LGPL: this unit compiles only with the opt-in
// transcription prepended (WOOF_GLIBC239_TRIG64, g239::g239_atan;
// flux_atmocn_e3sm.py refuses otherwise). log(zref/ztref) is folded by
// gfortran and arrives as an argument, folded on the host.
//
// gfortran's MAX(a, b) and MIN(a, b) keep a unless b is strictly greater
// (smaller); SIGN(a, b) is copysign(a, b).
//
// Attribute vectors are MCT's rAttr(nflds, lsize): field k of point n at
// av[n * nflds + k]. One thread per point.

#ifdef WOOF_GLIBC239_TRIG64

typedef double r8;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))

__device__ __forceinline__ r8 max_first(r8 a, r8 b) {
    return __longlong_as_double(b > a ? __double_as_longlong(b) : __double_as_longlong(a));
}
__device__ __forceinline__ r8 min_first(r8 a, r8 b) {
    return __longlong_as_double(b < a ? __double_as_longlong(b) : __double_as_longlong(a));
}

// shr_flux_mod's parameters
#define ZREF 10.0
#define ZTREF 2.0
#define DTAUMIN 0.01
#define ALPHA 1.4
#define MAXSCL 2.0
#define TD0 (-10.0)
#define SPVAL 1.0e30

// the flags
#define F_DEAD 1
#define F_IMPLICIT 2
#define F_GUST 4
#define F_COLDAIR 8

// the constants array: shr_flux_mod's loc_* (as shr_flux_adjust_constants
// leaves them), log(zref/ztref) folded, seq_flux_atmocn_minwind and
// flux_con_tol, in this order
enum { C_ZVIR, C_CPDAIR, C_CPVIR, C_KARMAN, C_G, C_LATVAP, C_STEBOL, C_AL2, C_MINWIND, C_TOL, NC };

// the fields read, as their columns in a2x / o2x (-1: not read)
enum { I_Z, I_U, I_V, I_WSRESP, I_TAU_EST, I_UGUST, I_PTEM, I_SHUM, I_DENS, I_TBOT, NA2X };
enum { O_T, O_U, O_V, O_FSWPEN, NO2X };
// the fields written, as their columns in xao
enum { X_SEN, X_LAT, X_TAUX, X_TAUY, X_EVAP, X_TREF, X_QREF, X_USTAR, X_RE, X_SSQ, X_LWUP, X_DUU10N,
       X_U10, X_U10G, X_FSWPEN, NXOUT };
// the scratch row per point: the outputs of shr_flux_atmOcn, then fswpen
enum { W_SEN, W_LAT, W_TAUX, W_TAUY, W_EVAP, W_TREF, W_QREF, W_USTAR, W_RE, W_SSQ, W_LWUP, W_DUU10N,
       W_U10RES, W_FSWPEN, NW };

// statement functions
__device__ __forceinline__ r8 qsat(r8 tk) { return D(640380.0, glibc_exp(D(5107.4, tk))); }
__device__ __forceinline__ r8 cdn(r8 umps) {
    return A(A(D(0.0027, umps), 0.000142), M(0.0000764, umps));
}
__device__ __forceinline__ r8 psimhu(r8 xd) {
    return A(S(glibc_log(D(M(A(1.0, M(xd, A(2.0, xd))), A(1.0, M(xd, xd))), 8.0)),
               M(2.0, g239::g239_atan(xd))), 1.571);
}
__device__ __forceinline__ r8 psixhu(r8 xd) { return M(2.0, glibc_log(D(A(1.0, M(xd, xd)), 2.0))); }
// x**4: libgcc's __powidf2 (y = 1; x = x*x; x = x*x; y = y*x)
__device__ __forceinline__ r8 pow4(r8 x) { const r8 x2 = M(x, x); return M(1.0, M(x2, x2)); }

// shr_flux_update_stress
__device__ __forceinline__ void update_stress(r8 wind0, r8 wsresp, r8 tau_est, r8* tau, r8* prev_tau,
                                              r8* tau_diff, r8* prev_tau_diff, r8* wind_adj)
{
    const r8 wsresp_applied = M(0.5, wsresp);
    if (M(S(*tau, tau_est), wsresp_applied) > M(0.99, wind0))
        *tau = A(tau_est, D(M(0.99, wind0), wsresp_applied));
    *prev_tau_diff = *tau_diff;
    *tau_diff = S(*tau, *prev_tau);
    const r8 tau_diff_fac = M(*tau_diff, *prev_tau_diff) < 0.0 ? 0.6 : 0.95;
    if (fabs(*tau_diff) > fabs(M(tau_diff_fac, *prev_tau_diff))) {
        *tau_diff = copysign(M(tau_diff_fac, *prev_tau_diff), *tau_diff);
        *tau = A(*prev_tau, *tau_diff);
    }
    *prev_tau = *tau;
    *wind_adj = S(wind0, M(S(*tau, tau_est), wsresp_applied));
}

// the wind the iteration sees after a stress update (gustiness, then the minimum)
__device__ __forceinline__ r8 gusted(r8 v, r8 ugust, int flags, r8 minwind) {
    if (flags & F_GUST) v = sqrt(A(M(v, v), M(ugust, ugust)));
    return max_first(minwind, v);
}

// seq_flux_init_mct: mask = 0 where the domain's mask rounds to 0 (< 0.5) or
// ofrac + ifrac <= 0, else 1
extern "C" __global__ void flux_init_mask(
    int* __restrict__ mask, const r8* __restrict__ dom, int ndom, int kmask, const r8* __restrict__ fr, int nfr,
    int ko, int ki, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int m = 1;
    if (dom[(size_t)i * ndom + kmask] < 0.5) m = 0;
    if (A(fr[(size_t)i * nfr + ko], fr[(size_t)i * nfr + ki]) <= 0.0) m = 0;
    mask[i] = m;
}

// seq_flux_atmocn_mct up to the write: gather (or the dead models'
// constants), emask, shr_flux_atmOcn. work[n * NW + w]; noiter[i] = 1 where
// a point did no iteration (shr_flux_atmOcn stops)
extern "C" __global__ void flux_atmocn(
    r8* __restrict__ work, int* __restrict__ noiter, int* __restrict__ mask,
    const r8* __restrict__ a2x, int na, const int* __restrict__ ka,
    const r8* __restrict__ o2x, int no, const int* __restrict__ ko,
    const r8* __restrict__ cst, int max_iter, int flags, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    struct { r8 zvir, cpdair, cpvir, karman, g, latvap, stebol, al2, minwind, tol; int max_iter; } c = {
        cst[C_ZVIR], cst[C_CPDAIR], cst[C_CPVIR], cst[C_KARMAN], cst[C_G], cst[C_LATVAP], cst[C_STEBOL],
        cst[C_AL2], cst[C_MINWIND], cst[C_TOL], max_iter};
    int emask = mask[i];
    r8 zbot, ubot, vbot, wsresp = 0.0, tau_est = 0.0, ugust = 0.0, thbot, qbot, rbot, tbot, ts, us, vs, fswpen;
    if (flags & F_DEAD) {
        mask[i] = 1;
        ts = 290.0; us = 0.0; vs = 0.0; zbot = 55.0; ubot = 0.0; vbot = 2.0;
        thbot = 301.0; qbot = 1.e-2; rbot = 1.0; tbot = 300.0; fswpen = 0.0;
    } else {
        if (!mask[i]) {
            noiter[i] = 0;
            return;
        }
        const r8* a = a2x + (size_t)i * na;
        const r8* o = o2x + (size_t)i * no;
        zbot = a[ka[I_Z]]; ubot = a[ka[I_U]]; vbot = a[ka[I_V]];
        if (flags & F_IMPLICIT) { wsresp = a[ka[I_WSRESP]]; tau_est = a[ka[I_TAU_EST]]; }
        if (flags & F_GUST) ugust = a[ka[I_UGUST]];
        thbot = a[ka[I_PTEM]]; qbot = a[ka[I_SHUM]]; rbot = a[ka[I_DENS]]; tbot = a[ka[I_TBOT]];
        ts = o[ko[O_T]]; us = o[ko[O_U]]; vs = o[ko[O_V]];
        // dens(n) < 1.0e-12: a default-real literal, float32(1e-12) widened
        if (rbot < (r8)1.0e-12f || ts < 1.0) emask = 0;
        fswpen = o[ko[O_FSWPEN]];
    }
    r8* w = work + (size_t)i * NW;
    w[W_FSWPEN] = fswpen;
    noiter[i] = 0;
    if (!emask) {
        for (int k = 0; k < W_FSWPEN; ++k) w[k] = SPVAL;
        return;
    }
    // shr_flux_atmOcn, ocn_surface_flux_scheme = 0, no wave coupling
    const r8 tdiff = S(tbot, ts);
    const r8 wind0 = max_first(sqrt(A(M(S(ubot, us), S(ubot, us)), M(S(vbot, vs), S(vbot, vs)))), 0.01);
    r8 vmag = gusted(wind0, ugust, flags, c.minwind);
    if ((flags & F_COLDAIR) && tdiff < TD0) {
        const r8 vscl = min_first(A(1.0, M(ALPHA, D(glibc_pow(fabs(S(tdiff, TD0)), 0.5), fabs(vmag)))), MAXSCL);
        vmag = M(vmag, vscl);
    }
    const r8 ssq = D(M(0.98, qsat(ts)), rbot);
    const r8 delt = S(thbot, ts);
    const r8 delq = S(qbot, ssq);
    const r8 alz = glibc_log(D(zbot, ZREF));
    const r8 cp = M(c.cpdair, A(1.0, M(c.cpvir, ssq)));
    r8 stable = A(0.5, copysign(0.5, delt));
    r8 rdn = sqrt(cdn(vmag));
    r8 rhn = A(M(S(1.0, stable), 0.0327), M(stable, 0.018));
    r8 ren = 0.0346;
    r8 ustar = M(rdn, vmag);
    r8 tstar = M(rhn, delt);
    r8 qstar = M(ren, delq);
    r8 ustar_prev = M(ustar, 2.0);
    r8 prev_tau = 0.0, prev_tau_diff = 0.0, tau = 0.0;
    if (flags & F_IMPLICIT) prev_tau = tau_est;
    r8 tau_diff = 1.e100;
    r8 wind_adj = wind0;
    if (flags & F_IMPLICIT) {
        tau = M(M(M(rbot, ustar), rdn), wind_adj);
        update_stress(wind0, wsresp, tau_est, &tau, &prev_tau, &tau_diff, &prev_tau_diff, &wind_adj);
        vmag = gusted(wind_adj, ugust, flags, c.minwind);
    } else {
        tau_diff = 0.0;
    }
    r8 hol = SPVAL, psixh = SPVAL, rd = 0.0, rh = SPVAL, re = 0.0, u10n = SPVAL;
    int iter = 0;
    while ((fabs(D(S(ustar, ustar_prev), ustar)) > c.tol || fabs(tau_diff) > DTAUMIN) && iter < c.max_iter) {
        iter = iter + 1;
        ustar_prev = ustar;
        hol = D(M(M(M(c.karman, c.g), zbot), A(D(tstar, thbot), D(qstar, A(D(1.0, c.zvir), qbot)))),
                M(ustar, ustar));
        hol = copysign(min_first(fabs(hol), 10.0), hol);
        stable = A(0.5, copysign(0.5, hol));
        const r8 xsq = max_first(sqrt(fabs(S(1.0, M(16.0, hol)))), 1.0);
        const r8 xqq = sqrt(xsq);
        const r8 psimh = A(M(M(-5.0, hol), stable), M(S(1.0, stable), psimhu(xqq)));
        psixh = A(M(M(-5.0, hol), stable), M(S(1.0, stable), psixhu(xqq)));
        rd = D(rdn, max_first(A(1.0, M(D(rdn, c.karman), S(alz, psimh))), 1.e-3));
        u10n = D(M(vmag, rd), rdn);
        rdn = sqrt(cdn(u10n));
        ren = 0.0346;
        rhn = A(M(S(1.0, stable), 0.0327), M(stable, 0.018));
        rd = D(rdn, max_first(A(1.0, M(D(rdn, c.karman), S(alz, psimh))), 1.e-3));
        rh = D(rhn, A(1.0, M(D(rhn, c.karman), S(alz, psixh))));
        re = D(ren, A(1.0, M(D(ren, c.karman), S(alz, psixh))));
        ustar = M(rd, vmag);
        tstar = M(rh, delt);
        qstar = M(re, delq);
        if (flags & F_IMPLICIT) {
            tau = M(M(M(rbot, ustar), rd), wind_adj);
            update_stress(wind0, wsresp, tau_est, &tau, &prev_tau, &tau_diff, &prev_tau_diff, &wind_adj);
            vmag = gusted(wind_adj, ugust, flags, c.minwind);
        }
    }
    if (iter < 1) {
        noiter[i] = 1;
        return;
    }
    tau = M(M(rbot, ustar), ustar);
    w[W_TAUX] = M(D(M(tau, S(ubot, us)), vmag), D(wind_adj, wind0));
    w[W_TAUY] = M(D(M(tau, S(vbot, vs)), vmag), D(wind_adj, wind0));
    w[W_SEN] = D(M(M(cp, tau), tstar), ustar);
    w[W_LAT] = D(M(M(c.latvap, tau), qstar), ustar);
    w[W_LWUP] = -M(c.stebol, pow4(ts));
    w[W_EVAP] = D(w[W_LAT], c.latvap);
    hol = D(M(hol, ZTREF), zbot);
    const r8 xsq = max_first(1.0, sqrt(fabs(S(1.0, M(16.0, hol)))));
    const r8 xqq = sqrt(xsq);
    const r8 psix2 = A(M(M(-5.0, hol), stable), M(S(1.0, stable), psixhu(xqq)));
    r8 fac = M(D(rh, c.karman), A(S(A(alz, c.al2), psixh), psix2));
    w[W_TREF] = S(S(thbot, M(delt, fac)), M(0.01, ZTREF));
    fac = M(D(re, c.karman), A(S(A(alz, c.al2), psixh), psix2));
    w[W_QREF] = S(qbot, M(delq, fac));
    w[W_DUU10N] = M(u10n, u10n);
    w[W_U10RES] = D(M(u10n, wind_adj), vmag);
    w[W_USTAR] = ustar;
    w[W_RE] = re;
    w[W_SSQ] = ssq;
}

// seq_flux_atmocn_mct's write into xao where mask /= 0; the dead models'
// diurnal fields are 0 (otherwise they are written back as read)
extern "C" __global__ void flux_atmocn_write(
    r8* __restrict__ xao, int nx, const int* __restrict__ kx, const int* __restrict__ kdiurnal, int ndiurnal,
    const r8* __restrict__ work, const int* __restrict__ mask, int dead, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !mask[i]) return;
    r8* x = xao + (size_t)i * nx;
    const r8* w = work + (size_t)i * NW;
    x[kx[X_SEN]] = w[W_SEN];
    x[kx[X_LAT]] = w[W_LAT];
    x[kx[X_TAUX]] = w[W_TAUX];
    x[kx[X_TAUY]] = w[W_TAUY];
    x[kx[X_EVAP]] = w[W_EVAP];
    x[kx[X_TREF]] = w[W_TREF];
    x[kx[X_QREF]] = w[W_QREF];
    x[kx[X_USTAR]] = w[W_USTAR];
    x[kx[X_RE]] = w[W_RE];
    x[kx[X_SSQ]] = w[W_SSQ];
    x[kx[X_LWUP]] = w[W_LWUP];
    x[kx[X_DUU10N]] = w[W_DUU10N];
    x[kx[X_U10]] = w[W_U10RES];
    x[kx[X_U10G]] = sqrt(w[W_DUU10N]);
    if (dead)
        for (int k = 0; k < ndiurnal; ++k) x[kdiurnal[k]] = 0.0;
    x[kx[X_FSWPEN]] = w[W_FSWPEN];
}

#endif  // WOOF_GLIBC239_TRIG64
