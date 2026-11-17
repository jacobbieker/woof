// RRTMG-K shortwave (WRF 4.7.1 ra_sw_physics = 14,
// phys/module_ra_rrtmg_swk.F): the routines whose arithmetic differs from
// the legacy RRTMG SW of kernels/rrtmg_sw.cu.  This file is appended to
// that unit (woof/core/rrtmgk.py, CudaSWK), so it uses its AD/SU/MU/DV
// macros, packed table offsets (RSW_T_*, RSW_C_*), rsw_etbl, RswStrided,
// rsw_vrtqdr and every unchanged kernel (inatm, setcoef, taumol,
// sfluxzen, the g-point accumulation, post).  The host packs K's own
// liquid tables (rrsw_cld_k extliq1/ssaliq1/asyliq1) into the
// RSW_T_CLD_*LIQ1 slots; every other table is word-for-word the legacy
// one.
//
// What K changes, statement by statement (module_ra_rrtmg_swk.F):
//   * cldprmc_sw (:1451-1898) keeps the legacy body and, in the cloudy
//     inflag >= 2 branch, stores the per-phase delta-scaled optical depth,
//     single-scattering albedo and scaled asymmetry
//     (dtX = tauX, dwX = ssaX, daX = (gX - forwX) / (1 - forwX)) before
//     the cldmin fix-ups.  rrtmg_sw sets them to 0 / 1 / 0 for every
//     g-point and layer before the call (:9111-9113); each thread here
//     writes those defaults for its own cell first.
//   * spcvmc_sw (:8024-8775) drops the gas/aerosol delta scaling, clamps
//     a non-positive ztaug to zero, and hands reftra_sw per-layer
//     two-stream coefficients (al1/al2/al3) built per phase from those
//     slabs, with zenith-dependent backscatter weights.
//   * reftra_sw (:1932-2244) runs kmodts = 5: gamma1..3 are al1..3,
//     gamma4 = zw - gamma3, and the reflectance/transmittance carry no
//     zw factor (kmodts >= 4).
// Aerosol: K's rrtmg_sw runs iaer = 6 and, with ecaer = 0 (the driver's
// aer_opt = 0), every band's ztaua/zasya/zomga is +0.  The one K
// statement with a transcendental, zf = pasya**1.5 (prmu0 < 0.5), is
// therefore pow(+-0, 1.5) = +0; a nonzero asymmetry is not transcribed
// and sets err_flag bit 16 (fail closed).

// ---------------------------------------------------------------------------
// cldprmc_sw (K).  One thread per (col, ig, lay); dtwa holds the nine
// (ncol, nlayers, 112) slabs dtliq, dtice, dtsno, dwliq, dwice, dwsno,
// daliq, daice, dasno, in that order.
// ---------------------------------------------------------------------------

#define RSWK_DT_LIQ 0
#define RSWK_DT_ICE 1
#define RSWK_DT_SNO 2
#define RSWK_DW_LIQ 3
#define RSWK_DW_ICE 4
#define RSWK_DW_SNO 5
#define RSWK_DA_LIQ 6
#define RSWK_DA_ICE 7
#define RSWK_DA_SNO 8
#define RSWK_NSLAB 9

__device__ void rswk_cldprmc_body(int ig, int lay,
                 int nlayers, int inflag, int iceflag, int liqflag,
                 const real* __restrict__ tab,
                 const int* __restrict__ ngb,
                 const real* __restrict__ cldfmc,
                 const real* __restrict__ ciwpmc,
                 const real* __restrict__ clwpmc,
                 const real* __restrict__ cswpmc,
                 const real* __restrict__ reicmc,
                 const real* __restrict__ relqmc,
                 const real* __restrict__ resnmc,
                 real* taormc, real* taucmc, real* ssacmc, real* asmcmc,
                 const real* __restrict__ fsfcmc,
                 real* dtwa, size_t slab,
                 int* err_flag)
{
    size_t p = (size_t)ig + (size_t)RSW_NGPT * lay;

    dtwa[RSWK_DT_LIQ * slab + p] = 0.0f;
    dtwa[RSWK_DT_ICE * slab + p] = 0.0f;
    dtwa[RSWK_DT_SNO * slab + p] = 0.0f;
    dtwa[RSWK_DW_LIQ * slab + p] = 1.0f;
    dtwa[RSWK_DW_ICE * slab + p] = 1.0f;
    dtwa[RSWK_DW_SNO * slab + p] = 1.0f;
    dtwa[RSWK_DA_LIQ * slab + p] = 0.0f;
    dtwa[RSWK_DA_ICE * slab + p] = 0.0f;
    dtwa[RSWK_DA_SNO * slab + p] = 0.0f;

    taormc[p] = taucmc[p];

    real cwp = AD(AD(ciwpmc[p], clwpmc[p]), cswpmc[p]);
    const real cldmin = 1.0e-20f;
    if (!(cldfmc[p] >= cldmin && (cwp >= cldmin || taucmc[p] >= cldmin)))
        return;

    // The K driver only reaches inflag 3..5, iceflag 5, liqflag 1: the
    // adapter admits snow-carrying microphysics only (iceflag 5).
    if (inflag < 2 || liqflag != 1 || iceflag != 5) {
        atomicOr(err_flag, 1);
        return;
    }

    real radice = reicmc[lay];
    real extcoice = 0.0f, ssacoice = 0.0f, gice = 0.0f, forwice = 0.0f;
    real extcosno = 0.0f, ssacosno = 0.0f, gsno = 0.0f, forwsno = 0.0f;
    int c = ngb[ig] - 16;

    if (EQ0(AD(ciwpmc[p], cswpmc[p]))) {
        // the ice coefficients stay zero (snow: its own test below)
    } else {   // iceflag >= 3
        if (radice < 5.0f || radice > 140.0f) { atomicOr(err_flag, 2); return; }
        real factor = DV(SU(radice, 2.0f), 3.0f);
        int index = (int)factor;
        if (index == 46) index = 45;
        real fint = SU(factor, (real)index);
        const real* e3 = tab + RSW_T_CLD_EXTICE3 + (size_t)46 * c;
        const real* s3 = tab + RSW_T_CLD_SSAICE3 + (size_t)46 * c;
        const real* a3 = tab + RSW_T_CLD_ASYICE3 + (size_t)46 * c;
        const real* f3 = tab + RSW_T_CLD_FDLICE3 + (size_t)46 * c;
        extcoice = AD(e3[index - 1], MU(fint, SU(e3[index], e3[index - 1])));
        ssacoice = AD(s3[index - 1], MU(fint, SU(s3[index], s3[index - 1])));
        gice = AD(a3[index - 1], MU(fint, SU(a3[index], a3[index - 1])));
        real fdelta = AD(f3[index - 1], MU(fint, SU(f3[index], f3[index - 1])));
        if (fdelta < 0.0f || fdelta > 1.0f) { atomicOr(err_flag, 2); return; }
        forwice = AD(fdelta, DV(0.5f, ssacoice));
        if (forwice > gice) forwice = gice;
        if (extcoice < 0.0f || ssacoice > 1.0f || ssacoice < 0.0f ||
            gice > 1.0f || gice < 0.0f) { atomicOr(err_flag, 2); return; }
    }

    if (cswpmc[p] > 0.0f) {   // iceflag == 5
        real radsno = resnmc[lay];
        if (radsno < 5.0f || radsno > 140.0f) { atomicOr(err_flag, 4); return; }
        real factor = DV(SU(radsno, 2.0f), 3.0f);
        int index = (int)factor;
        if (index == 46) index = 45;
        real fint = SU(factor, (real)index);
        const real* e3 = tab + RSW_T_CLD_EXTICE3 + (size_t)46 * c;
        const real* s3 = tab + RSW_T_CLD_SSAICE3 + (size_t)46 * c;
        const real* a3 = tab + RSW_T_CLD_ASYICE3 + (size_t)46 * c;
        const real* f3 = tab + RSW_T_CLD_FDLICE3 + (size_t)46 * c;
        extcosno = AD(e3[index - 1], MU(fint, SU(e3[index], e3[index - 1])));
        ssacosno = AD(s3[index - 1], MU(fint, SU(s3[index], s3[index - 1])));
        gsno = AD(a3[index - 1], MU(fint, SU(a3[index], a3[index - 1])));
        real fdelta = AD(f3[index - 1], MU(fint, SU(f3[index], f3[index - 1])));
        if (fdelta < 0.0f || fdelta > 1.0f) { atomicOr(err_flag, 4); return; }
        forwsno = AD(fdelta, DV(0.5f, ssacosno));
        if (forwsno > gsno) forwsno = gsno;
        if (extcosno < 0.0f || ssacosno > 1.0f || ssacosno < 0.0f ||
            gsno > 1.0f || gsno < 0.0f) { atomicOr(err_flag, 4); return; }
    }

    real extcoliq = 0.0f, ssacoliq = 0.0f, gliq = 0.0f, forwliq = 0.0f;
    if (!EQ0(clwpmc[p])) {   // liqflag == 1
        real radliq = relqmc[lay];
        if (radliq < 1.5f || radliq > 60.0f) { atomicOr(err_flag, 8); return; }
        int index = (int)SU(radliq, 1.5f);
        if (index == 0) index = 1;
        if (index == 58) index = 57;
        real fint = SU(SU(radliq, 1.5f), (real)index);
        const real* el = tab + RSW_T_CLD_EXTLIQ1 + (size_t)58 * c;
        const real* sl = tab + RSW_T_CLD_SSALIQ1 + (size_t)58 * c;
        const real* al = tab + RSW_T_CLD_ASYLIQ1 + (size_t)58 * c;
        extcoliq = AD(el[index - 1], MU(fint, SU(el[index], el[index - 1])));
        ssacoliq = AD(sl[index - 1], MU(fint, SU(sl[index], sl[index - 1])));
        if (fint < 0.0f && ssacoliq > 1.0f) ssacoliq = sl[index - 1];
        gliq = AD(al[index - 1], MU(fint, SU(al[index], al[index - 1])));
        forwliq = MU(gliq, gliq);
        if (extcoliq < 0.0f || ssacoliq > 1.0f || ssacoliq < 0.0f ||
            gliq > 1.0f || gliq < 0.0f) { atomicOr(err_flag, 8); return; }
    }

    // iceflag == 5
    real tauliqorig = MU(clwpmc[p], extcoliq);
    real tauiceorig = MU(ciwpmc[p], extcoice);
    real tausnoorig = MU(cswpmc[p], extcosno);
    taormc[p] = AD(AD(tauliqorig, tauiceorig), tausnoorig);
    real ssaliq = DV(MU(ssacoliq, SU(1.0f, forwliq)),
                     SU(1.0f, MU(forwliq, ssacoliq)));
    real tauliq = MU(SU(1.0f, MU(forwliq, ssacoliq)), tauliqorig);
    real ssaice = DV(MU(ssacoice, SU(1.0f, forwice)),
                     SU(1.0f, MU(forwice, ssacoice)));
    real tauice = MU(SU(1.0f, MU(forwice, ssacoice)), tauiceorig);
    real ssasno = DV(MU(ssacosno, SU(1.0f, forwsno)),
                     SU(1.0f, MU(forwsno, ssacosno)));
    real tausno = MU(SU(1.0f, MU(forwsno, ssacosno)), tausnoorig);
    real scatliq = MU(ssaliq, tauliq);
    real scatice = MU(ssaice, tauice);
    real scatsno = MU(ssasno, tausno);
    taucmc[p] = AD(AD(tauliq, tauice), tausno);

    dtwa[RSWK_DT_LIQ * slab + p] = tauliq;
    dtwa[RSWK_DW_LIQ * slab + p] = ssaliq;
    dtwa[RSWK_DT_ICE * slab + p] = tauice;
    dtwa[RSWK_DW_ICE * slab + p] = ssaice;
    dtwa[RSWK_DT_SNO * slab + p] = tausno;
    dtwa[RSWK_DW_SNO * slab + p] = ssasno;
    dtwa[RSWK_DA_LIQ * slab + p] = DV(SU(gliq, forwliq), SU(1.0f, forwliq));
    dtwa[RSWK_DA_ICE * slab + p] = DV(SU(gice, forwice), SU(1.0f, forwice));
    dtwa[RSWK_DA_SNO * slab + p] = DV(SU(gsno, forwsno), SU(1.0f, forwsno));

    if (EQ0(taucmc[p])) taucmc[p] = cldmin;
    if (EQ0(scatice)) scatice = cldmin;
    if (EQ0(scatsno)) scatsno = cldmin;

    ssacmc[p] = DV(AD(AD(scatliq, scatice), scatsno), taucmc[p]);
    asmcmc[p] = MU(DV(1.0f, AD(AD(scatliq, scatice), scatsno)),
        AD(AD(DV(MU(scatliq, SU(gliq, forwliq)), SU(1.0f, forwliq)),
              MU(scatice, DV(SU(gice, forwice), SU(1.0f, forwice)))),
           MU(scatsno, DV(SU(gsno, forwsno), SU(1.0f, forwsno)))));
}

extern "C" __global__
void rswk_cldprmc_b(int ncol, int nlayers,
                    int inflag, int iceflag, int liqflag,
                    const real* __restrict__ tab,
                    const int* __restrict__ ngb,
                    const real* __restrict__ cldfmc,   // (ncol, nl, 112)
                    const real* __restrict__ ciwpmc,
                    const real* __restrict__ clwpmc,
                    const real* __restrict__ cswpmc,
                    const real* __restrict__ reicmc,   // (ncol, nl)
                    const real* __restrict__ relqmc,
                    const real* __restrict__ resnmc,
                    real* taormc, real* taucmc, real* ssacmc, real* asmcmc,
                    const real* __restrict__ fsfcmc,
                    int* err_flag,
                    real* dtwa)                        // (9, ncol, nl, 112)
{
    long long W = (long long)nlayers * RSW_NGPT;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * W) return;
    int col = (int)(idx / W);
    int r = (int)(idx % W);
    int ig = r % RSW_NGPT;
    int lay = r / RSW_NGPT;
    size_t og = (size_t)col * nlayers * RSW_NGPT;
    size_t o = (size_t)col * nlayers;
    rswk_cldprmc_body(ig, lay, nlayers, inflag, iceflag, liqflag, tab, ngb,
                      cldfmc + og, ciwpmc + og, clwpmc + og, cswpmc + og,
                      reicmc + o, relqmc + o, resnmc + o,
                      taormc + og, taucmc + og, ssacmc + og, asmcmc + og,
                      fsfcmc + og, dtwa + og,
                      (size_t)ncol * nlayers * RSW_NGPT, err_flag);
}

// ---------------------------------------------------------------------------
// reftra_sw (K, kmodts = 5) for one layer.  Same structure, table idiom
// and order as rsw_reftra_layer; the differences are the gamma inputs
// (al1, al2, al3), gamma4 = zw - gamma3, and no zw in pref/ptra.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void rswk_reftra_layer(bool lrtchk,
                           real pgg, real prmuz, real ptau, real pw,
                           real al1, real al2, real al3,
                           const real* __restrict__ exp_tbl,
                           real& pref, real& prefd, real& ptra, real& ptrad)
{
    const real eps = 1.0e-08f;
    const real zwcrit = 0.9999995f;
    if (!lrtchk) {
        pref = 0.0f;
        ptra = 1.0f;
        prefd = 0.0f;
        ptrad = 1.0f;
        return;
    }
    real zto1 = ptau;
    real zw = pw;
    real zg = pgg;

    real zgamma1 = al1;
    real zgamma2 = al2;
    real zgamma3 = al3;
    real zgamma4 = SU(zw, zgamma3);

    real zwo = 0.0f;
    real denom = 1.0f;
    if (zg != 1.0f) {
        real q = DV(zg, SU(1.0f, zg));
        denom = SU(1.0f, MU(SU(1.0f, zw), MU(q, q)));
    }
    if (zw > 0.0f && denom != 0.0f) zwo = DV(zw, denom);

    if (zwo >= zwcrit) {
        real za = MU(zgamma1, prmuz);
        real za1 = SU(za, zgamma3);
        real zgt = MU(zgamma1, zto1);
        real ze1 = DV(zto1, prmuz);
        if (!(ze1 < 500.0f)) ze1 = 500.0f;   // min(x, 500)
        real ze2 = rsw_etbl(exp_tbl, ze1);
        pref = DV(SU(zgt, MU(za1, SU(1.0f, ze2))), AD(1.0f, zgt));
        ptra = SU(1.0f, pref);
        prefd = DV(zgt, AD(1.0f, zgt));
        ptrad = SU(1.0f, prefd);
        if (ze2 == 1.0f) {
            pref = 0.0f;
            ptra = 1.0f;
            prefd = 0.0f;
            ptrad = 1.0f;
        }
    } else {
        real za1 = AD(MU(zgamma1, zgamma4), MU(zgamma2, zgamma3));
        real za2 = AD(MU(zgamma1, zgamma3), MU(zgamma2, zgamma4));
        real zrk = __fsqrt_rn(SU(MU(zgamma1, zgamma1),
                                 MU(zgamma2, zgamma2)));
        real zrp = MU(zrk, prmuz);
        real zrp1 = AD(1.0f, zrp);
        real zrm1 = SU(1.0f, zrp);
        real zrk2 = MU(2.0f, zrk);
        real zrpp = SU(1.0f, MU(zrp, zrp));
        real zrkg = AD(zrk, zgamma1);
        real zr1 = MU(zrm1, AD(za2, MU(zrk, zgamma3)));
        real zr2 = MU(zrp1, SU(za2, MU(zrk, zgamma3)));
        real zr3 = MU(zrk2, SU(zgamma3, MU(za2, prmuz)));
        real zr4 = MU(zrpp, zrkg);
        real zr5 = MU(zrpp, SU(zrk, zgamma1));
        real zt1 = MU(zrp1, AD(za1, MU(zrk, zgamma4)));
        real zt2 = MU(zrm1, SU(za1, MU(zrk, zgamma4)));
        real zt3 = MU(zrk2, AD(zgamma4, MU(za1, prmuz)));
        real zt4 = zr4;
        real zt5 = zr5;
        real zbeta = DV(SU(zgamma1, zrk), zrkg);

        real ze1 = MU(zrk, zto1);
        if (!(ze1 < 500.0f)) ze1 = 500.0f;
        real ze2 = DV(zto1, prmuz);
        if (!(ze2 < 500.0f)) ze2 = 500.0f;
        real zem1, zep1, zem2, zep2;
        if (ze1 <= RSW_C_OD_LO) {
            zem1 = AD(SU(1.0f, ze1), MU(MU(0.5f, ze1), ze1));
            zep1 = DV(1.0f, zem1);
        } else {
            real tblind = DV(ze1, AD(RSW_C_BPADE, ze1));
            int itind = (int)AD(MU(RSW_C_TBLINT, tblind), 0.5f);
            zem1 = exp_tbl[itind];
            zep1 = DV(1.0f, zem1);
        }
        if (ze2 <= RSW_C_OD_LO) {
            zem2 = AD(SU(1.0f, ze2), MU(MU(0.5f, ze2), ze2));
            zep2 = DV(1.0f, zem2);
        } else {
            real tblind = DV(ze2, AD(RSW_C_BPADE, ze2));
            int itind = (int)AD(MU(RSW_C_TBLINT, tblind), 0.5f);
            zem2 = exp_tbl[itind];
            zep2 = DV(1.0f, zem2);
        }

        real zdenr = AD(MU(zr4, zep1), MU(zr5, zem1));
        real zdent = AD(MU(zt4, zep1), MU(zt5, zem1));
        if (zdenr >= -eps && zdenr <= eps) {
            pref = eps;
            ptra = zem2;
        } else {   // kmodts >= 4
            pref = DV(SU(SU(MU(zr1, zep1), MU(zr2, zem1)), MU(zr3, zem2)),
                      zdenr);
            ptra = SU(zem2,
                DV(MU(zem2, SU(SU(MU(zt1, zep1), MU(zt2, zem1)),
                               MU(zt3, zep2))), zdent));
        }

        real zemm = MU(zem1, zem1);
        real zdend = DV(1.0f, MU(SU(1.0f, MU(zbeta, zemm)), zrkg));
        prefd = MU(MU(zgamma2, SU(1.0f, zemm)), zdend);
        ptrad = MU(MU(zrk2, zem1), zdend);
    }
}

// zenith-weighted backscatter, 0.25 (2 - 3 g mu0) with the K overrides
__device__ __forceinline__ real rswk_b0mu(real w, real g, real prmu0)
{
    return MU(w, SU(2.0f, MU(MU(3.0f, g), prmu0)));
}

// ---------------------------------------------------------------------------
// spcvmc_sw (K) per g-point: the legacy rsw_spcvmc_body pipeline (same
// workspace, nodel products, layer combination and vrtqdr sweeps) with
// K's layer optics.  ua_* and us_* are Fortran variables equal to 2; the
// products 2 * x below are those statements with the 2 kept as an
// operand, in source order.
// ---------------------------------------------------------------------------
__device__ void rswk_spcvmc_body(int iw1, int nlayers,
                    const real* __restrict__ tab,
                    const int* __restrict__ ngb,
                    const real* __restrict__ palbd,    // (14)
                    const real* __restrict__ palbp,
                    const real* __restrict__ pcldfmc,  // (nl, 112)
                    const real* __restrict__ ptaucmc,
                    const real* __restrict__ pasycmc,
                    const real* __restrict__ pomgcmc,
                    const real* __restrict__ ptaormc,
                    const real* __restrict__ ptaua,    // (14, nl)
                    const real* __restrict__ pasya,
                    const real* __restrict__ pomga,
                    real prmu0,
                    const real* __restrict__ adjflux,  // (14)
                    const real* __restrict__ sfluxzen, // (112)
                    const real* __restrict__ taug,     // (nl, 112)
                    const real* __restrict__ taur,
                    const real* __restrict__ dtwa,     // 9 x (nl, 112)
                    size_t slab,
                    real* zincflx,                     // (112)
                    real* zcd, real* zcu,              // (nl+1, 112)
                    real* zfd, real* zfu,
                    real* ztdbt_nodel_out,
                    real* ztdbtc_nodel_out,
                    real* wk, size_t wks,
                    int* err_flag)
{
    int klev = nlayers;
    int ibm = ngb[iw1] - 15;                            // 1..14
    const real* exp_tbl = tab + RSW_T_EXP_TBL;
    const real repclc = 1.0e-12f;
    const real ua = 2.0f, us = 2.0f;

    const int n1 = nlayers + 1;
    const rsw_wk_t zrefc{wk + (size_t)0 * n1 * wks, wks};
    const rsw_wk_t zrefdc{wk + (size_t)1 * n1 * wks, wks};
    const rsw_wk_t ztrac{wk + (size_t)2 * n1 * wks, wks};
    const rsw_wk_t ztradc{wk + (size_t)3 * n1 * wks, wks};
    const rsw_wk_t zref{wk + (size_t)4 * n1 * wks, wks};
    const rsw_wk_t zrefd{wk + (size_t)5 * n1 * wks, wks};
    const rsw_wk_t ztra{wk + (size_t)6 * n1 * wks, wks};
    const rsw_wk_t ztrad{wk + (size_t)7 * n1 * wks, wks};
    const rsw_wk_t zdbtc{wk + (size_t)8 * n1 * wks, wks};
    const rsw_wk_t ztdbtc{wk + (size_t)9 * n1 * wks, wks};
    const rsw_wk_t zdbt{wk + (size_t)10 * n1 * wks, wks};
    const rsw_wk_t ztdbt{wk + (size_t)11 * n1 * wks, wks};
    const rsw_wk_t zrdnd{wk + (size_t)12 * n1 * wks, wks};
    const rsw_wk_t zrup{wk + (size_t)13 * n1 * wks, wks};
    const rsw_wk_t zrupd{wk + (size_t)14 * n1 * wks, wks};
    const rsw_wk_t ztdn{wk + (size_t)15 * n1 * wks, wks};
    const size_t os = RSW_NGPT;
    const size_t ob = iw1;
    const rsw_wk_t clear_fd{zcd + ob, os}, clear_fu{zcu + ob, os};
    const rsw_wk_t total_fd{zfd + ob, os}, total_fu{zfu + ob, os};
    const rsw_wk_t nodel_c{ztdbtc_nodel_out + ob, os};
    const rsw_wk_t nodel_t{ztdbt_nodel_out + ob, os};
    real ztdbtc_nodel = 1.0f;
    real ztdbt_nodel = 1.0f;
    nodel_c[0] = ztdbtc_nodel;
    nodel_t[0] = ztdbt_nodel;
    zincflx[iw1] = MU(MU(adjflux[ibm - 1], sfluxzen[iw1]), prmu0);

    ztdbtc[0] = 1.0f;
    zdbtc[klev] = 0.0f;
    ztrac[klev] = 0.0f;
    ztradc[klev] = 0.0f;
    zrefc[klev] = palbp[ibm - 1];
    zrefdc[klev] = palbd[ibm - 1];
    ztdbt[0] = 1.0f;
    zdbt[klev] = 0.0f;
    ztra[klev] = 0.0f;
    ztrad[klev] = 0.0f;
    zref[klev] = palbp[ibm - 1];
    zrefd[klev] = palbd[ibm - 1];
    zrup[klev] = palbp[ibm - 1];
    zrupd[klev] = palbd[ibm - 1];

    for (int J = 0; J < klev; ++J) {
        int jk = J + 1;
        int ikl = klev + 1 - jk;
        int K = ikl - 1;
        size_t pKw = (size_t)K * RSW_NGPT + iw1;
        size_t pKb = (size_t)K + (size_t)nlayers * (ibm - 1);
        bool lrtchkclr = 1;
        bool lrtchkcld = (pcldfmc[pKw] > repclc) ? 1 : 0;
        real tg = taug[pKw];
        if (tg <= 0.0f) tg = 0.0f;
        real tr = taur[pKw];
        real pa = pasya[pKb], po = pomga[pKb], pt = ptaua[pKb];

        real ztauc = AD(AD(tr, tg), pt);
        real zomcc = AD(MU(tr, 1.0f), MU(pt, po));
        real zgcc = DV(MU(MU(pa, po), pt), zomcc);
        zomcc = DV(zomcc, ztauc);

        real zf = MU(pa, pa);
        if (prmu0 < 0.5f) {
            // pasya**1.5: every K aerosol asymmetry is +-0 (iaer = 6,
            // ecaer = 0), and pow(+-0, 1.5) = +0.
            if (!EQ0(pa)) atomicOr(err_flag, 16);
            zf = 0.0f;
        }
        real zwf = MU(po, zf);
        real g_aer = DV(SU(pa, zf), SU(1.0f, zf));
        real tt_aer = MU(SU(1.0f, zwf), pt);
        real w_aer = DV(SU(po, zwf), SU(1.0f, zwf));
        real ts_aer = MU(tt_aer, w_aer);
        real ta_aer = MU(tt_aer, SU(1.0f, w_aer));
        real b0mu_aer = rswk_b0mu(0.25f, g_aer, prmu0);

        real dtl = dtwa[RSWK_DT_LIQ * slab + pKw];
        real dwl = dtwa[RSWK_DW_LIQ * slab + pKw];
        real dal = dtwa[RSWK_DA_LIQ * slab + pKw];
        real dti = dtwa[RSWK_DT_ICE * slab + pKw];
        real dwi = dtwa[RSWK_DW_ICE * slab + pKw];
        real dai = dtwa[RSWK_DA_ICE * slab + pKw];
        real dts = dtwa[RSWK_DT_SNO * slab + pKw];
        real dws = dtwa[RSWK_DW_SNO * slab + pKw];
        real das = dtwa[RSWK_DA_SNO * slab + pKw];
        real ts_liq = MU(dtl, dwl);
        real ta_liq = MU(dtl, SU(1.0f, dwl));
        real b0mu_liq = rswk_b0mu(0.25f, dal, prmu0);
        real b0_liq = MU(0.375f, SU(1.0f, dal));
        real ts_ice = MU(dti, dwi);
        real ta_ice = MU(dti, SU(1.0f, dwi));
        real b0mu_ice = rswk_b0mu(0.25f, dai, prmu0);
        real b0_ice = MU(0.375f, SU(1.0f, dai));
        real ts_sno = MU(dts, dws);
        real ta_sno = MU(dts, SU(1.0f, dws));
        real b0mu_sno = rswk_b0mu(0.25f, das, prmu0);
        real b0_sno = MU(0.375f, SU(1.0f, das));
        if (prmu0 < 0.3f)
            b0mu_aer = rswk_b0mu(0.35f, g_aer, prmu0);
        else if (prmu0 < 0.5f)
            b0mu_aer = rswk_b0mu(0.3f, g_aer, prmu0);
        else if (prmu0 > 0.7f)
            b0mu_aer = rswk_b0mu(0.22f, g_aer, prmu0);
        real b0_aer = MU(0.5f, SU(1.0f, g_aer));
        if (prmu0 < 0.5f)
            b0mu_liq = rswk_b0mu(0.3f, dal, prmu0);
        else if (prmu0 < 0.8f)
            b0mu_liq = rswk_b0mu(0.28f, dal, prmu0);
        if (prmu0 < 0.4f)
            b0mu_ice = rswk_b0mu(0.3f, dai, prmu0);
        else if (prmu0 > 0.9f)
            b0mu_ice = rswk_b0mu(0.18f, dai, prmu0);
        if (prmu0 < 0.4f)
            b0mu_sno = rswk_b0mu(0.3f, das, prmu0);

        real zclear = SU(1.0f, pcldfmc[pKw]);
        real zcloud = pcldfmc[pKw];
        {
            real ze1 = DV(ztauc, prmu0);
            real zdbtmc = rsw_etbl(exp_tbl, ze1);
            ztdbtc_nodel = MU(zdbtmc, ztdbtc_nodel);
            real tauorig = AD(ztauc, ptaormc[pKw]);
            ze1 = DV(tauorig, prmu0);
            real zdbtmo = rsw_etbl(exp_tbl, ze1);
            real zdbt_nodel = AD(MU(zclear, zdbtmc), MU(zcloud, zdbtmo));
            ztdbt_nodel = MU(zdbt_nodel, ztdbt_nodel);
        }
        nodel_c[J + 1] = ztdbtc_nodel;
        nodel_t[J + 1] = ztdbt_nodel;

        ztauc = AD(AD(tr, tg), tt_aer);
        zomcc = AD(MU(tr, 1.0f), MU(tt_aer, w_aer));
        zgcc = DV(MU(MU(g_aer, w_aer), tt_aer), zomcc);
        zomcc = DV(zomcc, ztauc);
        real tt = DV(1.0f, ztauc);
        real ray = MU(MU(us, 0.375f), tr);
        real aer_s = MU(MU(us, b0_aer), ts_aer);
        real al1c = MU(AD(AD(AD(MU(ua, tg), MU(ua, ta_aer)), ray), aer_s), tt);
        real al2c = MU(AD(ray, aer_s), tt);
        real al3c = MU(AD(MU(0.5f, tr), MU(b0mu_aer, ts_aer)), tt);

        // icpr = 1
        real ptc = ptaucmc[pKw];
        real ztauo = AD(ztauc, ptc);
        real zomco = AD(MU(ztauc, zomcc), MU(ptc, pomgcmc[pKw]));
        real zgco = DV(AD(MU(MU(ptc, pomgcmc[pKw]), pasycmc[pKw]),
                          MU(MU(ztauc, zomcc), zgcc)), zomco);
        zomco = DV(zomco, ztauo);
        tt = DV(1.0f, ztauo);
        real liq_s = MU(MU(us, b0_liq), ts_liq);
        real ice_s = MU(MU(us, b0_ice), ts_ice);
        real sno_s = MU(MU(us, b0_sno), ts_sno);
        real al1o = MU(AD(AD(AD(AD(AD(AD(AD(AD(AD(
                        MU(ua, tg), MU(ua, ta_aer)), MU(ua, ta_liq)),
                        MU(ua, ta_ice)), MU(ua, ta_sno)), ray), aer_s),
                        liq_s), ice_s), sno_s), tt);
        real al2o = MU(AD(AD(AD(AD(ray, aer_s), liq_s), ice_s), sno_s), tt);
        real al3o = MU(AD(AD(AD(AD(MU(0.5f, tr), MU(b0mu_aer, ts_aer)),
                                MU(b0mu_liq, ts_liq)), MU(b0mu_ice, ts_ice)),
                          MU(b0mu_sno, ts_sno)), tt);

        real zrefc_layer, zrefdc_layer, ztrac_layer, ztradc_layer;
        real zrefo, zrefdo, ztrao, ztrado;
        rswk_reftra_layer(lrtchkclr, zgcc, prmu0, ztauc, zomcc,
                          al1c, al2c, al3c, exp_tbl,
                          zrefc_layer, zrefdc_layer, ztrac_layer, ztradc_layer);
        rswk_reftra_layer(lrtchkcld, zgco, prmu0, ztauo, zomco,
                          al1o, al2o, al3o, exp_tbl,
                          zrefo, zrefdo, ztrao, ztrado);
        zrefc[J] = zrefc_layer;
        zrefdc[J] = zrefdc_layer;
        ztrac[J] = ztrac_layer;
        ztradc[J] = ztradc_layer;
        zref[J] = AD(MU(zclear, zrefc_layer), MU(zcloud, zrefo));
        zrefd[J] = AD(MU(zclear, zrefdc_layer), MU(zcloud, zrefdo));
        ztra[J] = AD(MU(zclear, ztrac_layer), MU(zcloud, ztrao));
        ztrad[J] = AD(MU(zclear, ztradc_layer), MU(zcloud, ztrado));

        real ze1 = DV(ztauc, prmu0);
        real zdbtmc = rsw_etbl(exp_tbl, ze1);
        zdbtc[J] = zdbtmc;
        ztdbtc[J + 1] = MU(zdbtc[J], ztdbtc[J]);
        ze1 = DV(ztauo, prmu0);
        real zdbtmo = rsw_etbl(exp_tbl, ze1);
        zdbt[J] = AD(MU(zclear, zdbtmc), MU(zcloud, zdbtmo));
        ztdbt[J + 1] = MU(zdbt[J], ztdbt[J]);
    }

    rsw_vrtqdr(klev, zrefc, zrefdc, ztrac, ztradc, zdbtc, zrdnd, zrup,
               zrupd, ztdbtc, clear_fd, clear_fu, ztdn);
    zrup[klev] = palbp[ibm - 1];
    zrupd[klev] = palbd[ibm - 1];
    rsw_vrtqdr(klev, zref, zrefd, ztra, ztrad, zdbt, zrdnd, zrup,
               zrupd, ztdbt, total_fd, total_fu, ztdn);
}

// Batched entry: rsw_spcvmc_gpt_b's argument list and thread mapping,
// plus the K slabs and the shared abort word.
extern "C" __global__
void rswk_spcvmc_gpt_b(int ncol, int nlayers,
                       const real* __restrict__ tab,
                       const int* __restrict__ ngb,
                       const real* __restrict__ palbd,    // (ncol, 14)
                       const real* __restrict__ palbp,
                       const real* __restrict__ pcldfmc,  // (ncol, nl, 112)
                       const real* __restrict__ ptaucmc,
                       const real* __restrict__ pasycmc,
                       const real* __restrict__ pomgcmc,
                       const real* __restrict__ ptaormc,
                       const real* __restrict__ ptaua,    // (ncol, 14, nl)
                       const real* __restrict__ pasya,
                       const real* __restrict__ pomga,
                       const real* __restrict__ prmu0,    // (ncol,)
                       const real* __restrict__ adjflux,  // (ncol, 14)
                       const real* __restrict__ sfluxzen, // (ncol, 112)
                       const real* __restrict__ taug,     // (ncol, nl, 112)
                       const real* __restrict__ taur,
                       real* zincflx,                     // (ncol, 112)
                       real* zcd, real* zcu,              // (ncol, n1, 112)
                       real* zfd, real* zfu,
                       real* ztdbt_nodel_out,
                       real* ztdbtc_nodel_out,
                       real* wk,           // (ncol*112, RSW_SPCVMC_WK * n1)
                       unsigned char* wkc,
                       const real* __restrict__ dtwa,     // (9, ncol, nl, 112)
                       int* err_flag)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * RSW_NGPT) return;
    int col = (int)(idx / RSW_NGPT);
    int iw1 = (int)(idx % RSW_NGPT);
    size_t n1 = (size_t)nlayers + 1;
    size_t ob = (size_t)col * RSW_NBND;
    size_t og = (size_t)col * RSW_NGPT;
    size_t ogl = (size_t)col * nlayers * RSW_NGPT;
    size_t obl = (size_t)col * nlayers * RSW_NBND;
    size_t og1 = (size_t)col * n1 * RSW_NGPT;
    rswk_spcvmc_body(iw1, nlayers, tab, ngb, palbd + ob, palbp + ob,
                     pcldfmc + ogl, ptaucmc + ogl, pasycmc + ogl,
                     pomgcmc + ogl, ptaormc + ogl,
                     ptaua + obl, pasya + obl, pomga + obl,
                     prmu0[col], adjflux + ob, sfluxzen + og,
                     taug + ogl, taur + ogl,
                     dtwa + ogl, (size_t)ncol * nlayers * RSW_NGPT,
                     zincflx + og,
                     zcd + og1, zcu + og1, zfd + og1, zfu + og1,
                     ztdbt_nodel_out + og1, ztdbtc_nodel_out + og1,
                     wk + (size_t)idx, (size_t)ncol * RSW_NGPT, err_flag);
}
