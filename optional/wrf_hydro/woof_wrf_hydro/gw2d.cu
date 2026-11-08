// WOOF-OPTIN: wrf-hydro
// Derived from WRF-Hydro (UCAR) as shipped in WRF 4.7.1 hydro/; see
// optional/wrf_hydro/LICENSE and NOTICE.md.  Not part of the default build or
// of any wheel.
//
// WRF-Hydro's 2-D groundwater model, GWBASESWCRT = 3
// (hydro/Routing/module_gw_gw2d.F90): gwSoilFlux with darcyGwSoil
// (:965-1134), gwstep (:154-913) as WRF-Hydro builds it, with MPP_LAND, on
// one process -- the two ADI sweeps are sub_tri_solv's (:2045-2127) over
// whole columns and rows, each reading the head as the previous stage left
// it -- and aggregateQsgw (:1138-1176).  float32 as WRF-Hydro, Fortran
// expressions in Fortran order, glibc's expf/powf (gfk_exp/gfk_pow,
// preceding this file).  Planes are (jx, ix), i fastest, as Fortran's
// (ix, jx).

#define GW_A(x, y) __fadd_rn((x), (y))
#define GW_S(x, y) __fsub_rn((x), (y))
#define GW_M(x, y) __fmul_rn((x), (y))
#define GW_D(x, y) __fdiv_rn((x), (y))

__device__ static float gw_min(float a, float b) { return a < b ? a : b; }   // MIN at -O0 (minss)

// darcyGwSoil (:1068-1134): the Boogaart et al. soil-groundwater flux, m/s
__device__ static float gw_darcy(float Z, float s, int soil)
{
    const float k_soil[12][9] = {
        {0.0778f, 3.9939f, 0.2913f, 4.0801f, 0.1386f, 4.0500f, -12.10f, 0.3950f, 1.0560f},
        {0.0924f, 4.8822f, 0.2674f, 3.8915f, 0.1365f, 4.3800f, -09.00f, 0.4100f, 0.9380f},
        {0.0367f, 4.5259f, 0.2446f, 4.2849f, 0.1208f, 4.9000f, -21.80f, 0.4350f, 0.2080f},
        {0.0101f, 3.6896f, 0.2153f, 4.2765f, 0.0887f, 5.3000f, -78.60f, 0.4850f, 0.0432f},
        {0.0101f, 3.6896f, 0.2153f, 4.2765f, 0.0887f, 5.3000f, -78.60f, 0.4850f, 0.0432f},
        {0.0169f, 2.9936f, 0.2858f, 4.3738f, 0.1026f, 5.3900f, -47.80f, 0.4510f, 0.0417f},
        {0.0271f, 4.4743f, 0.2587f, 3.9055f, 0.0920f, 7.1200f, -29.90f, 0.4200f, 0.0378f},
        {0.0227f, 4.3768f, 0.2658f, 3.8234f, 0.0843f, 7.7500f, -35.60f, 0.4770f, 0.0102f},
        {0.0127f, 6.6836f, 0.1725f, 3.7512f, 0.0703f, 8.5200f, -63.00f, 0.4760f, 0.0147f},
        {0.0530f, 9.2423f, 0.1859f, 3.3688f, 0.0728f, 10.400f, -15.30f, 0.4260f, 0.0130f},
        {0.0165f, 5.3972f, 0.2479f, 3.5549f, 0.0641f, 10.400f, -49.00f, 0.4920f, 0.0062f},
        {0.0200f, 6.0106f, 0.2474f, 3.4788f, 0.0622f, 11.400f, -40.50f, 0.4820f, 0.0077f}};
    const float* k = k_soil[soil - 1];
    float sig_a = GW_S(1.0f, gfk_exp(GW_M(-k[0], Z)));
    float sig_b = GW_M(k[1], gfk_pow(Z, k[2]));
    float sig_c = GW_M(k[3], gfk_exp(-gfk_pow(Z, k[4])));
    float y = GW_D(sig_a, GW_A(1.0f, gfk_exp(GW_M(sig_b, GW_S(s, sig_c)))));
    float b = k[5], ks = k[8], aep = -k[6];
    float c = GW_A(GW_M(2.0f, b), 3.0f);
    float q_grav = GW_M(-ks, gfk_pow(s, c));
    float beta = GW_A(2.0f, GW_D(3.0f, b));
    float alpha = GW_A(1.0f, GW_D(1.5f, GW_S(beta, 1.0f)));
    float q_cap = GW_M(GW_M(ks, alpha), gfk_pow(GW_D(aep, Z), beta));
    float q_darcy = GW_A(GW_M(y, q_cap), q_grav);
    return GW_M(q_darcy, GW_D(1.0f, 6000.0f));
}

// gwSoilFlux (:965-1066), one thread per routing cell
extern "C" __global__ void gw_soilflux(
    int n, int nsoil, float zsoil_bot, float dt,
    const float* __restrict__ elrt, const int* __restrict__ soiltyp,
    float* __restrict__ smcrt, const float* __restrict__ smcmaxrt,
    const float* __restrict__ sldpth, float* __restrict__ ho,
    const float* __restrict__ bot, const float* __restrict__ poros,
    float* __restrict__ qsgwrt, float* __restrict__ qdarcy)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    size_t last = (size_t)(nsoil - 1) * n + c;
    float smcrel = GW_D(smcrt[last], smcmaxrt[last]);
    float ztrans = GW_S(GW_A(elrt[c], zsoil_bot), ho[c]);
    ztrans = GW_M(ztrans, 100.0f);
    if (ztrans == 0.0f) ztrans = -5.0f;
    float qs = 0.0f, qd = 0.0f;
    int soil = soiltyp[c];
    if (ztrans > 0.0f && soil < 13) {
        qd = gw_darcy(ztrans, smcrel, soil);
        qs = qd;
        if (GW_M(GW_S(ho[c], bot[c]), poros[c]) < GW_M(qs, dt)) { qd = 0.0f; qs = 0.0f; }
    } else if (ztrans < 0.0f && soil < 13) {
        float zres = -ztrans;
        for (int k = nsoil - 1; k >= 0; --k) {
            size_t ik = (size_t)k * n + c;
            float th = GW_M(sldpth[k], 100.0f);
            if (zres >= th) {
                smcrt[ik] = smcmaxrt[ik];
                zres = GW_S(zres, GW_M(sldpth[k], 100.0f));
            } else if (zres > GW_M(0.5f, GW_M(sldpth[k], 100.0f))) {
                float frac = GW_D(zres, GW_M(sldpth[k], 100.0f));
                smcrt[ik] = GW_M(smcmaxrt[ik], frac);
            }
        }
    }
    qsgwrt[c] = qs;
    qdarcy[c] = qd;
    ho[c] = GW_A(ho[c], GW_D(GW_M(-qd, dt), poros[c]));   // headChange (:1062-1063)
}

// gwstep's transmissivities (:378-394, :564-580): t2 east, t1 north
extern "C" __global__ void gw_trans(
    int ix, int jx, float dx, const float* __restrict__ h,
    const float* __restrict__ elev, const float* __restrict__ bot,
    const float* __restrict__ hycond, float* __restrict__ t1, float* __restrict__ t2)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ix * jx) return;
    int i = c % ix, j = c / ix;
    int ip = i + 1 < ix ? i + 1 : ix - 1, jp = j + 1 < jx ? j + 1 : jx - 1;
    int ce = j * ix + ip, cn = jp * ix + i;
    float half = GW_M(0.5f, GW_A(dx, dx));
    float d0 = GW_S(gw_min(h[c], elev[c]), bot[c]);
    float de = GW_S(gw_min(h[ce], elev[ce]), bot[ce]);
    float dn = GW_S(gw_min(h[cn], elev[cn]), bot[cn]);
    t2[c] = GW_D(GW_M(__fsqrt_rn(fabsf(GW_M(GW_M(GW_M(hycond[c], d0), hycond[ce]), de))), half), half);
    t1[c] = GW_D(GW_M(__fsqrt_rn(fabsf(GW_M(GW_M(GW_M(hycond[c], d0), hycond[cn]), dn))), half), half);
}

// The N-S sweep (:504-552): for each column i, the system along j of
// (:510-532), solved by sub_tri_solv's ZDIR (:2088-2108); the neighbouring
// columns' heads are the stage's starting h.  One thread per column.
extern "C" __global__ void gw_sweep_ns(
    int ix, int jx, float dt, float darea, const float* __restrict__ sf2,
    const float* __restrict__ ho, const float* __restrict__ h,
    const float* __restrict__ t1, const float* __restrict__ t2,
    float* __restrict__ wk, float* __restrict__ x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ix) return;
    float zfac = 0.0f, cprev_b = 0.0f;
    for (int j = 0; j < jx; ++j) {
        int c = j * ix + i;
        float bb = GW_M(GW_D(sf2[c], dt), darea);
        float dd = GW_M(GW_D(GW_M(ho[c], sf2[c]), dt), darea);
        float aa = 0.0f, cc = 0.0f;
        if (j != 0) { aa = -t1[c - ix]; bb = GW_A(bb, t1[c - ix]); }
        if (j != jx - 1) { cc = -t1[c]; bb = GW_A(bb, t1[c]); }
        if (i != 0) { bb = GW_A(bb, t2[c - 1]); dd = GW_A(dd, GW_M(h[c - 1], t2[c - 1])); }
        if (i != ix - 1) { bb = GW_A(bb, t2[c]); dd = GW_A(dd, GW_M(h[c + 1], t2[c])); }
        if (j == 0) {
            zfac = GW_D(1.0f, bb);
            x[c] = GW_M(dd, zfac);
        } else {
            float w = GW_M(cprev_b, zfac);
            wk[c - ix] = w;
            zfac = GW_D(1.0f, GW_S(bb, GW_M(aa, w)));
            x[c] = GW_M(GW_S(dd, GW_M(aa, x[c - ix])), zfac);
        }
        cprev_b = cc;
    }
    for (int j = jx - 2; j >= 0; --j) {
        int c = j * ix + i;
        x[c] = GW_S(x[c], GW_M(wk[c], x[c + ix]));
    }
}

// The E-W sweep (:690-737): for each row j, the system along i of
// (:691-713), solved by sub_tri_solv's XDIR (:2063-2086); the neighbouring
// rows' heads are the N-S sweep's.  One thread per row.
extern "C" __global__ void gw_sweep_ew(
    int ix, int jx, float dt, float darea, const float* __restrict__ sf2,
    const float* __restrict__ ho, const float* __restrict__ h,
    const float* __restrict__ t1, const float* __restrict__ t2,
    float* __restrict__ wk, float* __restrict__ x)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= jx) return;
    float xfac = 0.0f, cprev_b = 0.0f;
    for (int i = 0; i < ix; ++i) {
        int c = j * ix + i;
        float bb = GW_M(GW_D(sf2[c], dt), darea);
        float dd = GW_M(GW_D(GW_M(ho[c], sf2[c]), dt), darea);
        float aa = 0.0f, cc = 0.0f;
        if (j != 0) { bb = GW_A(bb, t1[c - ix]); dd = GW_A(dd, GW_M(h[c - ix], t1[c - ix])); }
        if (j != jx - 1) { dd = GW_A(dd, GW_M(h[c + ix], t1[c])); bb = GW_A(bb, t1[c]); }
        if (i != 0) { bb = GW_A(bb, t2[c - 1]); aa = -t2[c - 1]; }
        if (i != ix - 1) { bb = GW_A(bb, t2[c]); cc = -t2[c]; }
        if (i == 0) {
            xfac = GW_D(1.0f, bb);
            x[c] = GW_M(dd, xfac);
        } else {
            float w = GW_M(cprev_b, xfac);
            wk[c - 1] = w;
            xfac = GW_D(1.0f, GW_S(bb, GW_M(aa, w)));
            x[c] = GW_M(GW_S(dd, GW_M(aa, x[c - 1])), xfac);
        }
        cprev_b = cc;
    }
    for (int i = ix - 2; i >= 0; --i) {
        int c = j * ix + i;
        x[c] = GW_S(x[c], GW_M(wk[c], x[c + 1]));
    }
}

// gwstep's land area (:357-366), the re-wetting fix and ocean exchange
// (:747-764), the surface excess and convergence (:801-816), and the
// rescaled diagnostics (:832-835): one thread, in Fortran's j-outer,
// i-inner order, because tareal, ebot and eocn are float sums.  e[0] is
// ebot, e[1] eocn.
extern "C" __global__ void gw_finish(
    int ix, int jx, float dt, float darea, const int* __restrict__ ltype,
    const float* __restrict__ sf2, const float* __restrict__ elev,
    const float* __restrict__ bot, const float* __restrict__ ho, float* __restrict__ h,
    float* __restrict__ convgw, float* __restrict__ excess, float* __restrict__ e)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float botinc = 0.01f, sealev = 1000.0f;
    float tareal = 0.0f, ebot = e[0], eocn = e[1];
    for (int c = 0; c < ix * jx; ++c)
        if (ltype[c] >= 1) tareal = GW_A(tareal, darea);
    for (int c = 0; c < ix * jx; ++c)
        if (ltype[c] == 1 && h[c] <= GW_A(bot[c], botinc)) {
            ebot = GW_A(ebot, GW_M(GW_M(GW_S(GW_A(bot[c], botinc), h[c]), sf2[c]), darea));
            h[c] = GW_A(bot[c], botinc);
        }
    for (int c = 0; c < ix * jx; ++c)
        if (ltype[c] == 2) eocn = GW_A(eocn, GW_M(GW_M(GW_S(h[c], sealev), sf2[c]), darea));
    for (int c = 0; c < ix * jx; ++c) {
        if (GW_S(elev[c], h[c]) < 0.0f) {
            excess[c] = GW_M(sf2[c], GW_S(h[c], elev[c]));
            h[c] = elev[c];
        } else {
            excess[c] = 0.0f;
        }
        convgw[c] = ltype[c] == 1 ? GW_D(GW_M(sf2[c], GW_S(h[c], ho[c])), dt) : 0.0f;
    }
    e[1] = GW_D(GW_D(eocn, tareal), dt);
    e[0] = GW_D(GW_D(ebot, tareal), dt);
}

// aggregateQsgw (:1138-1176), one thread per LSM cell
extern "C" __global__ void gw_aggregate(
    int ix, int jx, int agg, int ixrt, const float* __restrict__ qdarcy,
    float* __restrict__ qsgw)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ix * jx) return;
    int i = c % ix + 1, j = c / ix + 1;
    float s = 0.0f;
    for (int m = agg - 1; m >= 0; --m)
        for (int n = agg - 1; n >= 0; --n)
            s = GW_A(s, qdarcy[(j * agg - m - 1) * ixrt + (i * agg - n - 1)]);
    qsgw[c] = GW_D(s, (float)(agg * agg));
}

// HYDRO_drv's channel-aquifer head correction after drive_CHANNEL
// (module_HYDRO_drv.F90 :986-994, gwChanCondSw = 1): the diffusive wave's
// groundwater-channel flux (m3/s) leaves or enters the aquifer.
extern "C" __global__ void gw_chan_head(
    int n, float dt, float dx, const float* __restrict__ qgw_chanrt,
    const float* __restrict__ poros, float* __restrict__ ho)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    ho[c] = GW_A(ho[c], GW_D(GW_D(GW_M(GW_M(qgw_chanrt[c], -1.0f), dt), GW_M(dx, dx)), poros[c]));
}
