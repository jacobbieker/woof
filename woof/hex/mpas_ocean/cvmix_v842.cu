// MPAS-Ocean v8.4.2: CVMix vertical mixing coefficients (RKIND 8).
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_vmix_cvmix.F (ocn_vmix_coefs_cvmix_build,
//     ocn_vmix_cvmix_init)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// and of CVMix (CVMix/CVMix-src), the routines MPAS-Ocean v8.4.2 reaches:
//   cvmix_kpp.F90 (cvmix_init_kpp, cvmix_coeffs_kpp_low,
//     cvmix_kpp_compute_OBL_depth_low, cvmix_kpp_compute_kOBL_depth,
//     cvmix_kpp_compute_enhanced_diff, cvmix_kpp_compute_bulk_Richardson,
//     cvmix_kpp_compute_turbulent_scales_0d/_1d_sigma/_1d_OBL,
//     cvmix_kpp_compute_unresolved_shear, compute_phi_inv,
//     cvmix_kpp_compute_shape_function_coeffs)
//   cvmix_math.F90 (cvmix_math_poly_interp, cvmix_math_cubic_root_find,
//     cvmix_math_evaluate_cubic)
//   cvmix_shear.F90 (cvmix_coeffs_shear_low), cvmix_convection.F90
//     (cvmix_coeffs_conv_low)
// The CVMix developers (NCAR/UCAR staff and contributors; the sources carry
// no copyright line); Apache-2.0 WITH LLVM-exception, the
// terms in licenses/LICENSE-CVMix-Apache-2.0-LLVM.txt, as CVMix-src
// fce422195a0c58f15a55946b5ed517ba4365e232 is licensed. The words are held
// to v0.84-beta (d83f5827, the tag MPAS-Model v8.4.2 builds), whose routines
// on MPAS's path are those of the Apache-licensed tree (NOTICE, PROVENANCE).
// This is not the version available from the CVMix developers.
//
// RKIND 8 only (CVMix is real(8); mpas_ocn_vmix_cvmix.F does not compile at
// RKIND 4). Every operation is an explicit round-to-nearest intrinsic in
// gfortran -O0 -ffp-contract=off's order: x**2 is x*x, x**3 and a runtime
// integer power libgcc's __powidf2 (cv_powi), a real power glibc's pow
// (glibc_pow, glibc_flt64.cuh, prepended), sqrt the correctly rounded
// instruction. Fortran MIN/MAX are chosen on the words: the first argument
// unless the second compares less/greater. One thread runs one column, in the
// Fortran's loops; its column arrays live in `work` (1-based, as the
// Fortran's).
//
// Layout: a(nVertLevels[P1], nCells+1) is a[iCell*nl(+1) + k-1];
// normalVelocity(nVertLevels, nEdges+1) is u[iEdge*nl + k-1];
// cellsOnCell/edgesOnCell(maxEdges, nCells+1) is c[iCell*maxEdges + i-1],
// 1-based; cellMask(nVertLevels, nCells+1) is m[iCell*nl + k-1];
// vertNonLocalFlux(nGroup, nVertLevelsP1, nCells+1) is
// f[(iCell*(nl+1) + k-1)*nGroup + iGroup], iGroup = index_vertNonLocalFluxTemp - 1.

typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define SQRT(a) __dsqrt_rn(a)
__device__ __forceinline__ int cv_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }
// Fortran MIN/MAX on the words
__device__ __forceinline__ rk cv_min(rk a, rk b) { return b < a ? b : a; }
__device__ __forceinline__ rk cv_max(rk a, rk b) { return b > a ? b : a; }
// 1/3 as gfortran folds cvmix_one/real(3,cvmix_r8)
#define CV_THIRD 0.333333333333333314829616256247390992939472198486328125

// libgcc's __powidf2 (x**n for an integer n; gfortran -O0 calls it)
__device__ __forceinline__ rk cv_powi(rk x, int m)
{
    unsigned int n = m < 0 ? -(unsigned int)m : (unsigned int)m;
    rk y = (n % 2) ? x : 1.0;
    while (n >>= 1) {
        x = M(x, x);
        if (n % 2) y = M(y, x);
    }
    return m < 0 ? D(1.0, y) : y;
}

// cvmix_init_kpp's derived constants (MPAS passes ri_crit, the interpolation,
// lMonOb, the matching and surf_layer_ext; the rest are CVMix's defaults)
struct cv_kpp {
    rk vonkar, zeta_m, zeta_s, a_m, c_m, a_s, c_s, sle, ri_crit, nonlocal, minVtsqr;
};
__device__ __forceinline__ cv_kpp cv_init_kpp(rk sle, rk ri_crit)
{
    cv_kpp p;
    p.vonkar = 0.4;
    const rk cstar = 10.0, zm = -0.2, zs = -1.0;
    p.zeta_m = zm;
    p.zeta_s = zs;
    p.a_m = M(glibc_pow(S(1.0, M(16.0, zm)), -0.25), S(1.0, M(4.0, zm)));
    p.c_m = M(glibc_pow(S(1.0, M(16.0, zm)), -0.25), 12.0);
    p.a_s = M(SQRT(S(1.0, M(16.0, zs))), A(1.0, M(8.0, zs)));
    p.c_s = M(24.0, SQRT(S(1.0, M(16.0, zs))));
    p.sle = sle;
    p.ri_crit = ri_crit;
    p.nonlocal = M(M(cstar, p.vonkar), glibc_pow(M(M(p.vonkar, sle), p.c_s), CV_THIRD));
    p.minVtsqr = 1e-10;
    return p;
}

// compute_phi_inv
__device__ rk cv_phi_inv(rk zeta, bool scalar, const cv_kpp &p)
{
    if (zeta >= 0.0) return D(1.0, A(1.0, M(5.0, zeta)));
    if (!scalar) {
        if (zeta >= p.zeta_m) return glibc_pow(S(1.0, M(16.0, zeta)), 0.25);
        return glibc_pow(S(p.a_m, M(p.c_m, zeta)), CV_THIRD);
    }
    if (zeta >= p.zeta_s) return glibc_pow(S(1.0, M(16.0, zeta)), 0.5);
    return glibc_pow(S(p.a_s, M(p.c_s, zeta)), CV_THIRD);
}

// cvmix_kpp_compute_turbulent_scales_1d_sigma (w_m and w_s, l_LMD_ws false,
// no Langmuir factor); _0d is this with one sigma
__device__ void cv_scales_sigma(const rk *sigma, int n, rk obl, rk b, rk u, rk *wm, rk *ws, const cv_kpp &p)
{
    if (u != 0.0) {
        const rk u3 = cv_powi(u, 3);
        rk zprev = 0.0;
        for (int kw = 1; kw <= n; ++kw) {
            const rk zeta = D(M(M(M(cv_min(p.sle, sigma[kw]), obl), b), p.vonkar), u3);
            if (kw == 1) {
                wm[1] = M(M(cv_phi_inv(zeta, false, p), p.vonkar), u);
            } else if (zeta == zprev) {
                wm[kw] = wm[kw - 1];
            } else {
                wm[kw] = M(M(u, p.vonkar), cv_phi_inv(zeta, false, p));
            }
            zprev = zeta;
        }
        for (int kw = 1; kw <= n; ++kw) {
            const rk zeta = D(M(M(M(cv_min(p.sle, sigma[kw]), obl), b), p.vonkar), u3);
            if (kw == 1) {
                ws[1] = M(M(cv_phi_inv(zeta, true, p), p.vonkar), u);
            } else if (zeta == zprev) {
                ws[kw] = ws[kw - 1];
            } else {
                ws[kw] = M(M(u, p.vonkar), cv_phi_inv(zeta, true, p));
            }
            zprev = zeta;
        }
    } else if (b >= 0.0) {
        for (int kw = 1; kw <= n; ++kw) { wm[kw] = 0.0; ws[kw] = 0.0; }
    } else {
        for (int kw = 1; kw <= n; ++kw) {
            rk w = -M(M(M(M(p.c_m, cv_min(p.sle, sigma[kw])), obl), p.vonkar), b);
            wm[kw] = M(p.vonkar, glibc_pow(w, CV_THIRD));
        }
        for (int kw = 1; kw <= n; ++kw) {
            rk w = -M(M(M(M(p.c_s, cv_min(p.sle, sigma[kw])), obl), p.vonkar), b);
            ws[kw] = M(p.vonkar, glibc_pow(w, CV_THIRD));
        }
    }
}

__device__ __forceinline__ void cv_scales_0d(rk sigma, rk obl, rk b, rk u, rk *wm, rk *ws, const cv_kpp &p)
{
    rk s[2] = {0.0, sigma}, m[2], t[2];
    cv_scales_sigma(s, 1, obl, b, u, m, t, p);
    *wm = m[1];
    *ws = t[1];
}

// cvmix_math_poly_interp (the linear, quadratic and cubic-spline cases)
#define CV_LINEAR 1
#define CV_QUAD 2
#define CV_CUBE 3
__device__ void cv_poly_interp(rk *c, int type, rk x1, rk x2, rk y1, rk y2, bool has0, rk x0, rk y0)
{
    c[1] = 0.0; c[2] = 0.0; c[3] = 0.0; c[4] = 0.0;
    if (type == CV_LINEAR) {
        c[2] = D(S(y2, y1), S(x2, x1));
        c[1] = S(y1, M(c[2], x1));
    } else if (type == CV_QUAD) {
        const rk d = S(x2, x1);
        const rk det = -M(d, d);
        const rk rhs[4] = {0.0, y2, y1, has0 ? D(S(y1, y0), S(x1, x0)) : 0.0};
        rk mi[4][4];  // mi[row][col], 1-based
        mi[1][1] = -D(1.0, det);
        mi[1][2] = D(1.0, det);
        mi[1][3] = -D(1.0, S(x2, x1));
        mi[2][1] = D(M(2.0, x1), det);
        mi[2][2] = -D(M(2.0, x1), det);
        mi[2][3] = D(A(x2, x1), S(x2, x1));
        mi[3][1] = -D(M(x1, x1), det);
        mi[3][2] = D(M(x2, S(M(2.0, x1), x2)), det);
        mi[3][3] = -D(M(x2, x1), S(x2, x1));
        for (int k = 1; k <= 3; ++k)
            for (int k2 = 1; k2 <= 3; ++k2) c[k2] = A(c[k2], M(mi[4 - k2][k], rhs[k]));
    } else if (type == CV_CUBE) {
        const rk d = S(x2, x1);
        const rk det = -cv_powi(d, 3);
        const rk rhs[5] = {0.0, y2, y1, has0 ? D(S(y1, y0), S(x1, x0)) : 0.0, D(S(y2, y1), S(x2, x1))};
        rk mi[5][5];
        mi[1][1] = D(2.0, det);
        mi[1][2] = -D(2.0, det);
        mi[1][3] = D(S(x1, x2), det);
        mi[1][4] = D(S(x1, x2), det);
        mi[2][1] = -D(M(3.0, A(x2, x1)), det);
        mi[2][2] = D(M(3.0, A(x2, x1)), det);
        mi[2][3] = D(M(S(x2, x1), A(M(2.0, x2), x1)), det);
        mi[2][4] = D(M(S(x2, x1), A(M(2.0, x1), x2)), det);
        mi[3][1] = D(M(M(6.0, x2), x1), det);
        mi[3][2] = -D(M(M(6.0, x2), x1), det);
        mi[3][3] = -D(M(M(x2, S(x2, x1)), A(M(2.0, x1), x2)), det);
        mi[3][4] = -D(M(M(x1, S(x2, x1)), A(M(2.0, x2), x1)), det);
        mi[4][1] = -D(M(M(x1, x1), S(M(3.0, x2), x1)), det);
        mi[4][2] = -D(M(M(x2, x2), A(-M(3.0, x1), x2)), det);
        mi[4][3] = D(M(M(x1, M(x2, x2)), S(x2, x1)), det);
        mi[4][4] = D(M(M(x2, M(x1, x1)), S(x2, x1)), det);
        for (int k = 1; k <= 4; ++k)
            for (int k2 = 1; k2 <= 4; ++k2) c[k2] = A(c[k2], M(mi[5 - k2][k], rhs[k]));
    }
}

// cvmix_math_evaluate_cubic
__device__ __forceinline__ rk cv_eval_cubic(const rk *c, rk x)
{
    rk r = c[1];
    for (int i = 2; i <= 4; ++i) r = A(r, M(c[i], cv_powi(x, i - 1)));
    return r;
}

// cvmix_math_cubic_root_find (Newton, 100 iterations, |f| < 1e-12)
__device__ rk cv_cubic_root(const rk *c, rk x0)
{
    rk root = x0;
    rk f = A(A(A(M(c[4], cv_powi(root, 3)), M(c[3], M(root, root))), M(c[2], root)), c[1]);
    for (int it = 1; it <= 100; ++it) {
        if (fabs(f) < 1e-12) break;
        const rk slope = A(A(M(M(3.0, c[4]), M(root, root)), M(M(2.0, c[3]), root)), c[2]);
        root = S(root, D(f, slope));
        f = A(A(A(M(c[4], cv_powi(root, 3)), M(c[3], M(root, root))), M(c[2], root)), c[1]);
    }
    return root;
}

// cvmix_kpp_compute_kOBL_depth (zw(1:nlev+1), zt(1:nlev), 1-based)
__device__ rk cv_kobl(const rk *zw, const rk *zt, int nlev, rk obl)
{
    for (int kw = 1; kw <= nlev; ++kw)
        if (obl < fabs(zw[kw + 1])) return obl < fabs(zt[kw]) ? (rk)kw + 0.25 : (rk)kw + 0.75;
    return (rk)nlev + 0.75;
}

// the column's z: zw_iface(1:nl+1), zt_cntr(1:nl) as ocn_vmix_coefs_cvmix_build builds them
__device__ void cv_geometry(const rk *h, int mlc, int nl, rk *zw, rk *zt)
{
    zw[1] = 0.0;
    zt[1] = -D(h[0], 2.0);
    for (int k = 2; k <= mlc; ++k) {
        zw[k] = S(zw[k - 1], h[k - 2]);
        zt[k] = S(zw[k], D(h[k - 1], 2.0));
    }
    zw[mlc + 1] = S(zw[mlc], h[mlc - 1]);
    for (int k = mlc + 1; k <= nl; ++k) {
        zw[k + 1] = zw[mlc + 1];
        zt[k] = zw[mlc + 1];
    }
}

// real parameters (rp) and switches (ip)
enum { RP_SLE, RP_RICRIT, RP_FIXED_DEPTH, RP_MIN_ICE, RP_RHO_SW, RP_BG_VISC, RP_BG_DIFF, RP_PP_NU0, RP_PP_ALPHA,
       RP_PP_EXP, RP_KPP_NU0, RP_KPP_RI0, RP_KPP_EXP, RP_CONV_VISC, RP_CONV_DIFF, RP_CONV_TRIGGER, RP_N };
enum { IP_KPP, IP_FIXED, IP_INTERP, IP_MONOB, IP_MATCH_BOTH, IP_SHEAR, IP_CONV, IP_RI_LOOPS, IP_N };
#define CV_GRAVITY 9.80616
// per-column work: real arrays of nl+2 words
enum { W_ZW, W_ZT, W_BVF, W_NSQR, W_OBL, W_WS, W_NVSUM, W_PDSUM, W_DV2, W_RIS, W_RIT, W_M, W_T, W_TNL, W_OBLM,
       W_OBLT, W_SIGMA, W_WM, W_WSG, W_N };

// Stage 1 (kpp_stage == 1) of one column, cells 1..nCellsArray(3), KPP on:
// the boundary-layer depth, from the bulk Richardson numbers (or fixed), and
// its limits; writes bulkRichardsonNumber(Buoy/Shear) and boundaryLayerDepth.
extern "C" __global__ void ocn_cvmix_kpp_stage1_v842(
    rk *bulkRi, rk *bulkRiBuoy, rk *bulkRiShear, rk *boundaryLayerDepth, rk *work, int *iwork,
    const rk *layerThickness, const rk *normalVelocity, const rk *potentialDensity, const rk *bruntVaisalaFreqTop,
    const rk *surfaceFrictionVelocity, const rk *surfaceBuoyancyForcing, const rk *iceFraction, const rk *areaCell,
    const rk *dcEdge, const rk *dvEdge, const int *nEdgesOnCell, const int *edgesOnCell, const int *maxLevelCell,
    int nCells, int nl, int maxEdges, const rk *rp, const int *ip)
{
    const int iCell = cv_thread();
    if (iCell >= nCells) return;
    const int mlc = maxLevelCell[iCell];
    const int L = nl + 2;
    rk *w = work + (size_t)iCell * W_N * L;
    rk *zw = w + W_ZW * L, *zt = w + W_ZT * L, *bvfs = w + W_BVF * L, *nsqr = w + W_NSQR * L;
    rk *obld = w + W_OBL * L, *wsc = w + W_WS * L, *nvsum = w + W_NVSUM * L, *pdsum = w + W_PDSUM * L;
    rk *dv2 = w + W_DV2 * L;
    int *sai = iwork + (size_t)iCell * L;
    const rk *h = layerThickness + (size_t)iCell * nl;
    const rk *pd = potentialDensity + (size_t)iCell * nl;
    const rk *bvf = bruntVaisalaFreqTop + (size_t)iCell * nl;
    rk *ri = bulkRi + (size_t)iCell * nl, *rib = bulkRiBuoy + (size_t)iCell * nl, *ris = bulkRiShear + (size_t)iCell * nl;
    const cv_kpp p = cv_init_kpp(rp[RP_SLE], rp[RP_RICRIT]);
    const rk invAreaCell = D(1.0, areaCell[iCell]);
    const rk ustar = surfaceFrictionVelocity[iCell], bforce = surfaceBuoyancyForcing[iCell];

    cv_geometry(h, mlc, nl, zw, zt);
    for (int k = 1; k <= mlc; ++k) bvfs[k] = cv_max(0.0, bvf[k - 1]);
    bvfs[mlc + 1] = cv_max(0.0, bvfs[mlc]);

    rk bld;
    if (ip[IP_FIXED]) {
        bld = rp[RP_FIXED_DEPTH];
    } else {
        for (int k = 1; k <= mlc; ++k) nsqr[k] = bvfs[k];
        {
            const int k = mlc + 1 < nl ? mlc + 1 : nl;
            const rk v = nsqr[k - 1];
            for (int kk = k; kk <= mlc + 1; ++kk) nsqr[kk] = v;
        }
        int topIndex = 1;
        for (int kOBL = 1; kOBL <= mlc; ++kOBL) {
            dv2[kOBL] = 0.0;
            const rk depth = fabs(zw[kOBL + 1]);
            obld[kOBL] = fabs(zw[kOBL + 1]);
            sai[kOBL] = 1;
            const rk sfc = M(depth, p.sle);
            for (int kav = topIndex; kav <= kOBL; ++kav)
                if (zw[kav + 1] < -sfc) { sai[kOBL] = kav; break; }
            topIndex = sai[kOBL] - 1 > 1 ? sai[kOBL] - 1 : 1;
        }
        // cvmix_kpp_compute_turbulent_scales_1d_OBL (w_s; sigma_coord = surf_layer_ext)
        if (ustar != 0.0) {
            const rk u3 = cv_powi(ustar, 3);
            rk zprev = 0.0;
            for (int kw = 1; kw <= mlc; ++kw) {
                const rk zeta = D(M(M(M(cv_min(p.sle, p.sle), obld[kw]), bforce), p.vonkar), u3);
                if (kw > 1 && zeta == zprev) wsc[kw] = wsc[kw - 1];
                else wsc[kw] = M(M(cv_phi_inv(zeta, true, p), p.vonkar), ustar);
                zprev = zeta;
            }
        } else {
            for (int kw = 1; kw <= mlc; ++kw) {
                if (bforce >= 0.0) {
                    wsc[kw] = 0.0;
                } else {
                    const rk v = -M(M(M(M(p.c_s, cv_min(p.sle, p.sle)), obld[kw]), p.vonkar), bforce);
                    wsc[kw] = M(p.vonkar, glibc_pow(v, CV_THIRD));
                }
            }
        }
        // deltaVelocitySquared, the surface layer's mean velocity against each level's
        for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
            const int iEdge = edgesOnCell[(size_t)iCell * maxEdges + i] - 1;
            const rk *u = normalVelocity + (size_t)iEdge * nl;
            const rk factor = M(M(M(0.5, dcEdge[iEdge]), dvEdge[iEdge]), invAreaCell);
            dv2[1] = 0.0;
            nvsum[1] = u[0];
            for (int k = 2; k <= mlc; ++k) nvsum[k] = A(nvsum[k - 1], u[k - 1]);
            for (int k = 1; k <= mlc; ++k) {
                const rk av = D(nvsum[sai[k]], (rk)sai[k]);
                const rk du = S(av, u[k - 1]);
                dv2[k] = A(dv2[k], M(factor, M(du, du)));
            }
        }
        pdsum[1] = pd[0];
        for (int k = 2; k <= mlc; ++k) pdsum[k] = A(pdsum[k - 1], pd[k - 1]);
        for (int k = 1; k <= mlc; ++k) {
            ris[k - 1] = cv_max(dv2[k], 1.0e-15);
            rib[k - 1] = D(M(CV_GRAVITY, S(pd[k - 1], D(pdsum[sai[k]], (rk)sai[k]))), rp[RP_RHO_SW]);
        }
        // cvmix_kpp_compute_bulk_Richardson with cvmix_kpp_compute_unresolved_shear (Nsqr_iface)
        const rk vtc = D(SQRT(D(0.2, M(p.c_s, p.sle))), M(p.vonkar, p.vonkar));
        const rk scaling = S(1.0, M(0.5, p.sle));
        for (int kt = 1; kt <= mlc; ++kt) {
            const rk n = SQRT(cv_max(nsqr[kt + 1], 0.0));
            const rk cv = n < 0.002 ? S(2.1, M(200.0, n)) : 1.7;
            rk vt2 = -D(M(M(M(M(cv, vtc), zt[kt]), n), wsc[kt]), p.ri_crit);
            if (vt2 < p.minVtsqr) vt2 = p.minVtsqr;
            const rk num = -M(M(scaling, zt[kt]), rib[kt - 1]);
            const rk denom = A(ris[kt - 1], vt2);
            ri[kt - 1] = denom != 0.0 ? D(num, denom) : M(num, 1e10);
        }
        // cvmix_kpp_compute_OBL_depth_low (Ri_bulk at the centres: depth => zt)
        rk limit = fabs(zt[mlc]);
        if (ip[IP_MONOB]) {
            const rk mo = bforce > 0.0 ? D(cv_powi(ustar, 3), M(bforce, p.vonkar)) : fabs(zt[mlc]);
            limit = cv_min(limit, mo);
        }
        int k = 0;
        for (; k <= mlc - 1; ++k)
            if (ri[k] > p.ri_crit) break;
        if (k == mlc) {
            bld = fabs(limit);
        } else if (k == 0) {
            bld = fabs(zt[1]);
        } else {
            rk c[5];
            if (k == 1) cv_poly_interp(c, ip[IP_INTERP], zt[1], zt[2], ri[0], ri[1], false, 0.0, 0.0);
            else cv_poly_interp(c, ip[IP_INTERP], zt[k], zt[k + 1], ri[k - 1], ri[k], true, zt[k - 1], ri[k - 2]);
            c[1] = S(c[1], p.ri_crit);
            bld = -cv_cubic_root(c, M(0.5, A(zt[k], zt[k + 1])));
            if (k == 1) bld = cv_max(bld, -zt[1]);
            bld = cv_min(bld, limit);
        }
        bld = cv_max(bld, 0.0);  // minOBLdepth; maxOBLdepth = 0 is not applied
    }
    // the limits: half the top layer, the minimum under sea ice, the column's last centre
    const rk half = D(h[0], 2.0);
    if (bld < half) bld = half;
    if (iceFraction[iCell] > 0.15)
        if (bld < rp[RP_MIN_ICE]) bld = rp[RP_MIN_ICE];
    if (bld > fabs(zt[mlc])) bld = fabs(zt[mlc]);
    boundaryLayerDepth[iCell] = bld;
}

// config_cvmix_use_BLD_smoothing, cells 1..nCellsArray(2): the area-weighted
// mean of the column's and its masked neighbours' boundary-layer depths
// (boundaryLayerDepthSmooth; the copy back follows)
extern "C" __global__ void ocn_cvmix_bld_smooth_v842(
    rk *boundaryLayerDepthSmooth, const rk *boundaryLayerDepth, const int *nEdgesOnCell, const int *cellsOnCell,
    const int *cellMask, const rk *areaCell, int nCells, int nl, int maxEdges)
{
    const int iCell = cv_thread();
    if (iCell >= nCells) return;
    rk smooth = 0.0, areaSum = 0.0;
    int edgeCount = 0;
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const int n = cellsOnCell[(size_t)iCell * maxEdges + i] - 1;
        const int mask = cellMask[(size_t)n * nl];
        smooth = A(smooth, M(M(M(2.0, (rk)mask), areaCell[n]), boundaryLayerDepth[n]));
        areaSum = A(areaSum, M(M(2.0, areaCell[n]), (rk)mask));
        edgeCount += mask;
    }
    areaSum = A(areaSum, M((rk)edgeCount, areaCell[iCell]));
    smooth = A(smooth, M(M(boundaryLayerDepth[iCell], (rk)edgeCount), areaCell[iCell]));
    boundaryLayerDepthSmooth[iCell] = D(smooth, areaSum);
}

// Stage 2 (kpp_stage == 2) of one column, cells 1..nCellsArray(3): the
// smoothed Richardson number, shear mixing, KPP (cvmix_coeffs_kpp_low and the
// non-local flux), convection, the no-flux boundaries. The viscosity and
// diffusivity enter zeroed with the background added.
extern "C" __global__ void ocn_cvmix_stage2_v842(
    rk *vertViscTopOfCell, rk *vertDiffTopOfCell, rk *vertNonLocalFlux, rk *boundaryLayerDepth,
    rk *indexBoundaryLayerDepth, rk *work, const rk *layerThickness, const rk *riTopOfCell,
    const rk *bruntVaisalaFreqTop, const rk *surfaceFrictionVelocity, const rk *surfaceBuoyancyForcing,
    const int *maxLevelCell, int nCells, int nl, int nGroup, int iGroup, const rk *rp, const int *ip)
{
    const int iCell = cv_thread();
    if (iCell >= nCells) return;
    const int mlc = maxLevelCell[iCell];
    const int L = nl + 2;
    rk *w = work + (size_t)iCell * W_N * L;
    rk *zw = w + W_ZW * L, *zt = w + W_ZT * L, *bvfs = w + W_BVF * L, *ris = w + W_RIS * L, *rit = w + W_RIT * L;
    rk *Md = w + W_M * L, *Td = w + W_T * L, *tnl = w + W_TNL * L, *oblm = w + W_OBLM * L, *oblt = w + W_OBLT * L;
    rk *sigma = w + W_SIGMA * L, *wm = w + W_WM * L, *wsg = w + W_WSG * L;
    const rk *h = layerThickness + (size_t)iCell * nl;
    const rk *ritop = riTopOfCell + (size_t)iCell * (nl + 1);
    const rk *bvf = bruntVaisalaFreqTop + (size_t)iCell * nl;
    rk *visc = vertViscTopOfCell + (size_t)iCell * (nl + 1), *diff = vertDiffTopOfCell + (size_t)iCell * (nl + 1);
    const rk ustar = surfaceFrictionVelocity[iCell], bforce = surfaceBuoyancyForcing[iCell];

    cv_geometry(h, mlc, nl, zw, zt);
    // the 1-2-1 filter on RiTopOfCell, Jacobi sweeps
    for (int k = 1; k <= mlc; ++k) ris[k] = ritop[k - 1];
    ris[mlc + 1] = ris[mlc];
    for (int k = 1; k <= mlc + 1; ++k) rit[k] = ris[k];
    for (int n = 1; n <= ip[IP_RI_LOOPS]; ++n) {
        for (int k = 2; k <= mlc; ++k) ris[k] = D(A(A(rit[k - 1], M(2.0, rit[k])), rit[k + 1]), 4.0);
        for (int k = 1; k <= mlc; ++k) rit[k] = ris[k];
    }
    for (int k = 1; k <= mlc; ++k) bvfs[k] = cv_max(0.0, bvf[k - 1]);
    bvfs[mlc + 1] = cv_max(0.0, bvfs[mlc]);

    // shear mixing (cvmix_coeffs_shear_low), the background taken back out of PP's
    if (ip[IP_SHEAR]) {
        for (int kw = 1; kw <= mlc + 1; ++kw) {
            const rk r = ris[kw];
            if (ip[IP_SHEAR] == 1) {
                const rk denom = r > 0.0 ? A(1.0, M(rp[RP_PP_ALPHA], r)) : 1.0;
                Md[kw] = A(D(rp[RP_PP_NU0], glibc_pow(denom, rp[RP_PP_EXP])), 0.0);
                Td[kw] = A(D(Md[kw], denom), 0.0);
            } else {
                rk t;
                if (r < 0.0) {
                    t = rp[RP_KPP_NU0];
                } else if (r < rp[RP_KPP_RI0]) {
                    const rk q = D(r, rp[RP_KPP_RI0]);
                    t = M(rp[RP_KPP_NU0], glibc_pow(S(1.0, M(q, q)), rp[RP_KPP_EXP]));
                } else {
                    t = 0.0;
                }
                Td[kw] = t;
                Md[kw] = t;
            }
        }
        if (ip[IP_SHEAR] == 1)
            for (int k = 1; k <= mlc + 1; ++k) {
                Md[k] = S(Md[k], rp[RP_BG_VISC]);
                Td[k] = S(Td[k], rp[RP_BG_DIFF]);
            }
        for (int k = 1; k <= mlc; ++k) {
            visc[k - 1] = A(visc[k - 1], Md[k]);
            diff[k - 1] = A(diff[k - 1], Td[k]);
        }
    }

    // KPP: the boundary layer's diffusivities and the non-local flux
    rk kobl = 0.0;  // cvmix_variables%kOBL_depth, never set without KPP
    if (ip[IP_KPP]) {
        const cv_kpp p = cv_init_kpp(rp[RP_SLE], rp[RP_RICRIT]);
        for (int k = 1; k <= mlc + 1; ++k) { Md[k] = visc[k - 1]; Td[k] = diff[k - 1]; }
        const rk blTemp = cv_max(boundaryLayerDepth[iCell], D(h[0], 2.0));
        const rk obl = cv_min(blTemp, fabs(zt[mlc]));
        boundaryLayerDepth[iCell] = obl;
        kobl = cv_kobl(zw, zt, mlc, obl);
        // cvmix_coeffs_kpp_low (nlev = maxLevelCell; Mdiff_out is old_Mdiff, Md)
        const bool lstable = bforce > 0.0;
        const int kwup = (int)floor(kobl);
        const int ktup = (int)floor(kobl + 0.5) - 1;
        const rk delta = ktup == mlc ? D(A(obl, zt[ktup]), S(zt[ktup], zw[ktup + 1]))
                                     : D(A(obl, zt[ktup]), S(zt[ktup], zt[ktup + 1]));
        rk mshape[5], tshape[5];
        if (!ip[IP_MATCH_BOTH]) {
            mshape[1] = 0.0; mshape[2] = 1.0; mshape[3] = -2.0; mshape[4] = 1.0;
            for (int i = 1; i <= 4; ++i) tshape[i] = mshape[i];
        } else {
            rk wmObl, wsObl, c[5];
            cv_scales_0d(1.0, obl, bforce, ustar, &wmObl, &wsObl, p);
            if (kwup == 1) cv_poly_interp(c, ip[IP_INTERP], zw[1], zw[2], Md[1], Md[2], false, 0.0, 0.0);
            else cv_poly_interp(c, ip[IP_INTERP], zw[kwup], zw[kwup + 1], Md[kwup], Md[kwup + 1], true, zw[kwup - 1],
                                Md[kwup - 1]);
            const rk mObl = cv_eval_cubic(c, -obl);
            if (kwup == 1) cv_poly_interp(c, ip[IP_INTERP], zw[1], zw[2], Td[1], Td[2], false, 0.0, 0.0);
            else cv_poly_interp(c, ip[IP_INTERP], zw[kwup], zw[kwup + 1], Td[kwup], Td[kwup + 1], true, zw[kwup - 1],
                                Td[kwup - 1]);
            const rk tObl = cv_eval_cubic(c, -obl);
            rk mAt1 = 0.0, tAt1 = 0.0;
            if (obl != 0.0) {
                mAt1 = wmObl != 0.0 ? D(mObl, M(wmObl, obl)) : 0.0;
                tAt1 = wsObl != 0.0 ? D(tObl, M(wsObl, obl)) : 0.0;
            }
            // cvmix_kpp_compute_shape_function_coeffs with G'(1) = 0 (lnoDGat1)
            mshape[1] = 0.0; mshape[2] = 1.0;
            mshape[3] = S(S(M(3.0, mAt1), 0.0), 2.0);
            mshape[4] = A(S(0.0, M(2.0, mAt1)), 1.0);
            tshape[1] = 0.0; tshape[2] = 1.0;
            tshape[3] = S(S(M(3.0, tAt1), 0.0), 2.0);
            tshape[4] = A(S(0.0, M(2.0, tAt1)), 1.0);
        }
        for (int k = 1; k <= nl + 1; ++k) tnl[k] = 0.0;
        for (int k = 1; k <= ktup + 1; ++k) { oblm[k] = 0.0; oblt[k] = 0.0; }
        for (int kw = 1; kw <= mlc + 1; ++kw) sigma[kw] = -D(zw[kw], obl);
        cv_scales_sigma(sigma, mlc + 1, obl, bforce, ustar, wm, wsg, p);
        for (int kw = 2; kw <= kwup; ++kw) {
            const rk mS = cv_eval_cubic(mshape, sigma[kw]);
            const rk tS = cv_eval_cubic(tshape, sigma[kw]);
            if (!lstable) tnl[kw] = M(p.nonlocal, cv_eval_cubic(tshape, sigma[kw]));
            oblm[kw] = M(M(obl, wm[kw]), mS);
            oblt[kw] = M(M(obl, wsg[kw]), tS);
        }
        const rk sigmaKtup = -D(zt[ktup], obl);
        const rk mS = cv_eval_cubic(mshape, sigmaKtup);
        const rk tS = cv_eval_cubic(tshape, sigmaKtup);
        rk wmK, wsK;
        cv_scales_0d(sigmaKtup, obl, bforce, ustar, &wmK, &wsK, p);
        const rk mKtup = M(M(obl, wmK), mS);
        const rk tKtup = M(M(obl, wsK), tS);
        // cvmix_kpp_compute_enhanced_diff (ktup is kwup or kwup - 1)
        {
            const int j = ktup + 1;
            const rk omd = S(1.0, delta);
            const rk oldT = oblt[j];
            const rk omd2 = cv_powi(omd, 2), delta2 = cv_powi(delta, 2);
            if (ktup == kwup) {
                const rk enhM = A(M(omd2, mKtup), M(delta2, Md[j]));
                const rk enhT = A(M(omd2, tKtup), M(delta2, Td[j]));
                Md[j] = A(M(omd, Md[j]), M(delta, enhM));
                Td[j] = A(M(omd, Td[j]), M(delta, enhT));
                oblm[j] = Md[j];
                oblt[j] = Td[j];
            } else {
                const rk enhM = A(M(omd2, mKtup), M(delta2, oblm[j]));
                const rk enhT = A(M(omd2, tKtup), M(delta2, oblt[j]));
                oblm[j] = A(M(omd, Md[j]), M(delta, enhM));
                oblt[j] = A(M(omd, Td[j]), M(delta, enhT));
                tnl[j] = oldT != 0.0 ? D(M(tnl[j], oblt[j]), oldT) : 0.0;
            }
        }
        for (int k = 2; k <= ktup + 1; ++k) { Md[k] = oblm[k]; Td[k] = oblt[k]; }
        indexBoundaryLayerDepth[iCell] = kobl;
        if (!ip[IP_MATCH_BOTH]) {
            const int kb = (int)kobl;
            for (int k = 1; k <= kb; ++k) {
                visc[k - 1] = A(visc[k - 1], Md[k]);
                diff[k - 1] = A(diff[k - 1], Td[k]);
            }
            for (int k = kb + 1; k <= mlc + 1; ++k) { visc[k - 1] = Md[k]; diff[k - 1] = Td[k]; }
        } else {
            for (int k = 1; k <= mlc + 1; ++k) { visc[k - 1] = Md[k]; diff[k - 1] = Td[k]; }
        }
        rk *f = vertNonLocalFlux + (size_t)iCell * (nl + 1) * nGroup + iGroup;
        for (int k = 1; k <= nl + 1; ++k) f[(size_t)(k - 1) * nGroup] = tnl[k];
    }

    // convection (cvmix_coeffs_conv_low, Brunt-Vaisala; lnoOBL)
    if (ip[IP_CONV]) {
        const rk trig = rp[RP_CONV_TRIGGER];
        if (trig < 0.0) {
            for (int kw = 1; kw <= mlc; ++kw) {
                rk wgt = 0.0;
                if (bvfs[kw] <= 0.0) {
                    if (bvfs[kw] > trig) {
                        wgt = S(1.0, D(bvfs[kw], trig));
                        wgt = cv_powi(S(1.0, M(wgt, wgt)), 3);
                    } else {
                        wgt = 1.0;
                    }
                }
                Md[kw] = M(wgt, rp[RP_CONV_VISC]);
                Td[kw] = M(wgt, rp[RP_CONV_DIFF]);
            }
        } else {
            const int oblInd = (int)floor(kobl + 0.5) + 1;
            for (int kw = 1; kw <= mlc; ++kw) {
                const bool on = bvfs[kw] <= 0.0 && kw >= oblInd;
                Md[kw] = on ? rp[RP_CONV_VISC] : 0.0;
                Td[kw] = on ? rp[RP_CONV_DIFF] : 0.0;
            }
        }
        Md[mlc + 1] = 0.0;
        Td[mlc + 1] = 0.0;
        const int k0 = ip[IP_KPP] ? (int)indexBoundaryLayerDepth[iCell] + 1 : 1;
        const int k1 = ip[IP_KPP] ? mlc : mlc + 1;
        for (int k = k0; k <= k1; ++k) {
            visc[k - 1] = A(visc[k - 1], Md[k]);
            diff[k - 1] = A(diff[k - 1], Td[k]);
        }
    }

    // no-flux boundaries
    visc[0] = 0.0;
    diff[0] = 0.0;
    for (int k = mlc + 1; k <= nl + 1; ++k) { visc[k - 1] = 0.0; diff[k - 1] = 0.0; }
}

// ocn_vmix_implicit with config_use_cvmix: the viscosity averaged to the
// edges 1..nEdgesArray(1), zero below maxLevelEdgeTop
extern "C" __global__ void ocn_cvmix_visc_edge_v842(
    rk *vertViscTopOfEdge, const rk *vertViscTopOfCell, const int *cellsOnEdge, const int *maxLevelEdgeTop,
    int nEdges, int nl)
{
    const int t = cv_thread();
    if (t >= nEdges * (nl + 1)) return;
    const int iEdge = t / (nl + 1), k = t % (nl + 1);
    const int c1 = cellsOnEdge[2 * iEdge] - 1, c2 = cellsOnEdge[2 * iEdge + 1] - 1;
    vertViscTopOfEdge[t] = k < maxLevelEdgeTop[iEdge]
        ? M(0.5, A(vertViscTopOfCell[(size_t)c2 * (nl + 1) + k], vertViscTopOfCell[(size_t)c1 * (nl + 1) + k]))
        : 0.0;
}
