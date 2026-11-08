// WOOF-OPTIN: wrf-hydro
// Derived from WRF-Hydro (UCAR) as shipped in WRF 4.6.1 hydro/; see
// optional/wrf_hydro/LICENSE and NOTICE.md.  Not part of the default build or
// of any wheel.
//
// WRF-Hydro's reach channel routing, hydro/Routing/module_channel_routing.F90:
// MUSKING (:43-66), SUBMUSKINGCUNGE (:150-539) and the reach branch of
// drive_CHANNEL (channel_option 1 and 2, :704-865), without MPP_LAND.  The
// lateral inflow is a REAL*8 sum per reach over its pixels in pixel order and
// the upstream inflows are REAL sums over upstream reaches in reach order,
// both through CSR lists built once on the host.  Every reach in a substep
// reads the previous substep's QLINK, as WRF's do (k) loop does, so reaches
// run in parallel.  Every ** with a real exponent is glibc's powf (the -O0
// reference makes 16 powf calls, none folded), x**2 with an integer
// exponent is x*x.

__device__ float mc_musking(float qup, float quc, float qdp, float dt, float Km, float X)
{
    float dth = __fdiv_rn(dt, 3600.0f);
    float den = __fadd_rn(__fmul_rn(__fmul_rn(2.0f, Km), __fsub_rn(1.0f, X)), dth);
    float C1 = __fdiv_rn(__fsub_rn(dth, __fmul_rn(__fmul_rn(2.0f, Km), X)), den);
    float C2 = __fdiv_rn(__fadd_rn(dth, __fmul_rn(__fmul_rn(2.0f, Km), X)), den);
    float C3 = __fdiv_rn(__fsub_rn(__fmul_rn(__fmul_rn(2.0f, Km), __fsub_rn(1.0f, X)), dth), den);
    return __fadd_rn(__fadd_rn(__fmul_rn(C1, quc), __fmul_rn(C2, qup)), __fmul_rn(C3, qdp));
}

// AREA, WP, WPk, AREAC, WPC, R at depth hx (:279-302 and :352-375).
__device__ static void mc_geometry(float hx, float Bw, float bfd, float z, float TwCC,
                                   float nCC, float modK, float& AREA, float& AREAC,
                                   float& WP, float& WPC, float& WPk, float& R)
{
    WPC = 0.0f;
    AREAC = 0.0f;
    float s1z = __fsqrt_rn(__fadd_rn(1.0f, __fmul_rn(z, z)));
    if ((hx > bfd) && (TwCC > 0.0f) && (nCC > 0.0f)) {
        AREA = __fmul_rn(__fadd_rn(Bw, __fmul_rn(bfd, z)), bfd);
        AREAC = __fmul_rn(TwCC, __fsub_rn(hx, bfd));
        WP = __fadd_rn(Bw, __fmul_rn(__fmul_rn(2.0f, bfd), s1z));
        WPC = __fadd_rn(TwCC, __fmul_rn(2.0f, __fsub_rn(hx, bfd)));
        WPk = __fadd_rn(WP, __fmul_rn(WPC, fminf(__fdiv_rn(hx, __fmul_rn(modK, __fsqrt_rn(Bw))), 1.0f)));
        R = __fdiv_rn(__fadd_rn(AREA, AREAC), __fadd_rn(WP, WPC));
    } else {
        AREA = __fmul_rn(__fadd_rn(Bw, __fmul_rn(hx, z)), hx);
        WP = __fadd_rn(Bw, __fmul_rn(__fmul_rn(2.0f, hx), s1z));
        WPk = __fmul_rn(WP, fminf(__fdiv_rn(hx, __fmul_rn(modK, __fsqrt_rn(Bw))), 1.0f));
        if (WP > 0.0f) R = __fdiv_rn(AREA, WP);
        else R = 0.0f;
    }
}

// Ck at depth hx (:304-315 and :377-388).
__device__ static float mc_celerity(float hx, float Bw, float bfd, float z, float TwCC,
                                    float nCC, float n, float So, float AREA, float AREAC, float R)
{
    const float p23 = 2.0f / 3.0f, p53 = 5.0f / 3.0f;
    float s1z = __fsqrt_rn(__fadd_rn(1.0f, __fmul_rn(z, z)));
    float sqS = __fsqrt_rn(So);
    if ((hx > bfd) && (TwCC > 0.0f) && (nCC > 0.0f)) {
        float in = __fsub_rn(__fmul_rn(p53, gfk_pow(R, p23)),
                   __fmul_rn(__fmul_rn(p23, gfk_pow(R, p53)),
                             __fdiv_rn(__fmul_rn(2.0f, s1z),
                                       __fadd_rn(Bw, __fmul_rn(__fmul_rn(2.0f, bfd), z)))));
        float t1 = __fmul_rn(__fmul_rn(__fdiv_rn(sqS, n), in), AREA);
        float t2 = __fmul_rn(__fmul_rn(__fmul_rn(__fdiv_rn(sqS, nCC), p53),
                                       gfk_pow(__fsub_rn(hx, bfd), p23)), AREAC);
        return fmaxf(0.0f, __fdiv_rn(__fadd_rn(t1, t2), __fadd_rn(AREA, AREAC)));
    }
    if (hx > 0.0f) {
        float in = __fsub_rn(__fmul_rn(p53, gfk_pow(R, p23)),
                   __fmul_rn(__fmul_rn(p23, gfk_pow(R, p53)),
                             __fdiv_rn(__fmul_rn(2.0f, s1z),
                                       __fadd_rn(Bw, __fmul_rn(__fmul_rn(2.0f, hx), z)))));
        return fmaxf(0.0f, __fmul_rn(__fdiv_rn(sqS, n), in));
    }
    return 0.0f;
}

__device__ static float mc_qmn(float WP, float WPC, float n, float nCC, float AREA,
                               float AREAC, float R, float So)
{
    float nbar = __fdiv_rn(__fadd_rn(__fmul_rn(WP, n), __fmul_rn(WPC, nCC)), __fadd_rn(WP, WPC));
    return __fmul_rn(__fmul_rn(__fmul_rn(__fdiv_rn(1.0f, nbar), __fadd_rn(AREA, AREAC)),
                               gfk_pow(R, 2.0f / 3.0f)), __fsqrt_rn(So));
}

// SUBMUSKINGCUNGE; returns 1 on WRF's fatal channel-coefficient stop.
__device__ int mc_cunge(float& qdc, float& vel, float& qloss, float qup, float quc,
                        float qdp, float ql, float dt, float So, float dx, float n,
                        float Cs, float Bw, float Tw, float TwCC, float nCC,
                        float& depth, float ChannK, int& nonconv)
{
    float C1 = 0.0f, C2 = 0.0f, C3 = 0.0f, C4 = 0.0f, Km, X, Ck, Twl;
    float AREA = 0.0f, AREAC = 0.0f, WP = 0.0f, WPC = 0.0f, R = 0.0f, WPk = 0.0f;
    float h, h_0, h_1, bfd, Qj_0, Qj = 0.0f, D = 0.0f, aerror, rerror, z;
    const float modK = 0.15f, mindepth = 0.01f;
    int iter, maxiter = 100, tries = 0;
    aerror = 0.01f;
    rerror = 1.0f;
    if (Cs == 0.0f) z = 1.0f;
    else z = __fdiv_rn(1.0f, Cs);
    if (Bw > Tw) bfd = __fdiv_rn(Bw, 0.00001f);
    else if (Bw == Tw) bfd = __fdiv_rn(Bw, __fmul_rn(2.0f, z));
    else bfd = __fdiv_rn(__fsub_rn(Tw, Bw), __fmul_rn(2.0f, z));
    if (n <= 0.0f || So <= 0.0f || z <= 0.0f || Bw <= 0.0f) return 1;
    depth = fmaxf(depth, 0.0f);
    h = __fadd_rn(__fmul_rn(depth, 1.33f), mindepth);
    h_0 = __fmul_rn(depth, 0.67f);
    if (ql > 0.0f || qup > 0.0f || qdp > 0.0f) {
        for (;;) {   // label 110
            Qj_0 = 0.0f;
            iter = 0;
            while (rerror > 0.01f && aerror >= mindepth && iter <= maxiter) {
                Twl = __fadd_rn(Bw, __fmul_rn(__fmul_rn(2.0f, z), h_0));
                mc_geometry(h_0, Bw, bfd, z, TwCC, nCC, modK, AREA, AREAC, WP, WPC, WPk, R);
                Ck = mc_celerity(h_0, Bw, bfd, z, TwCC, nCC, n, So, AREA, AREAC, R);
                if (Ck > 0.0f) Km = fmaxf(dt, __fdiv_rn(dx, Ck));
                else Km = dt;
                if ((h_0 > bfd) && (TwCC > 0.0f) && (nCC > 0.0f) && (Ck > 0.0f))
                    X = fminf(0.5f, fmaxf(0.0f, __fmul_rn(0.5f, __fsub_rn(1.0f, __fdiv_rn(Qj_0,
                        __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(2.0f, TwCC), So), Ck), dx))))));
                else if (Ck > 0.0f)
                    X = fminf(0.5f, fmaxf(0.0f, __fmul_rn(0.5f, __fsub_rn(1.0f, __fdiv_rn(Qj_0,
                        __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(2.0f, Twl), So), Ck), dx))))));
                else
                    X = 0.5f;
                D = __fadd_rn(__fmul_rn(Km, __fsub_rn(1.000f, X)), __fdiv_rn(dt, 2.0000f));
                if (D == 0.0f) return 2;
                C1 = __fdiv_rn(__fadd_rn(__fmul_rn(Km, X), __fdiv_rn(dt, 2.000000f)), D);
                C2 = __fdiv_rn(__fsub_rn(__fdiv_rn(dt, 2.0000f), __fmul_rn(Km, X)), D);
                C3 = __fdiv_rn(__fsub_rn(__fmul_rn(Km, __fsub_rn(1.00000000f, X)), __fdiv_rn(dt, 2.000000f)), D);
                if (ChannK <= 0.0f) C4 = __fdiv_rn(__fmul_rn(ql, dt), D);
                else C4 = __fdiv_rn(__fmul_rn(__fsub_rn(ql, __fmul_rn(__fmul_rn(ChannK, dx), WPk)), dt), D);
                if (__fadd_rn(WP, WPC) > 0.0f) {
                    float s = __fadd_rn(__fadd_rn(__fmul_rn(C1, qup), __fmul_rn(C2, quc)), __fmul_rn(C3, qdp));
                    if ((C4 < 0.0f) && (fabsf(C4) > s)) C4 = -s;
                    Qj_0 = __fsub_rn(__fadd_rn(s, C4), mc_qmn(WP, WPC, n, nCC, AREA, AREAC, R, So));
                }
                Twl = __fadd_rn(Bw, __fmul_rn(__fmul_rn(2.0f, z), h));
                mc_geometry(h, Bw, bfd, z, TwCC, nCC, modK, AREA, AREAC, WP, WPC, WPk, R);
                Ck = mc_celerity(h, Bw, bfd, z, TwCC, nCC, n, So, AREA, AREAC, R);
                if (Ck > 0.0f) Km = fmaxf(dt, __fdiv_rn(dx, Ck));
                else Km = dt;
                {
                    float s = __fadd_rn(__fadd_rn(__fmul_rn(C1, qup), __fmul_rn(C2, quc)), __fmul_rn(C3, qdp));
                    float sc = __fadd_rn(s, C4);
                    if ((h > bfd) && (TwCC > 0.0f) && (nCC > 0.0f) && (Ck > 0.0f))
                        X = fminf(0.5f, fmaxf(0.25f, __fmul_rn(0.5f, __fsub_rn(1.0f, __fdiv_rn(sc,
                            __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(2.0f, TwCC), So), Ck), dx))))));
                    else if (Ck > 0.0f)
                        X = fminf(0.5f, fmaxf(0.25f, __fmul_rn(0.5f, __fsub_rn(1.0f, __fdiv_rn(sc,
                            __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(2.0f, Twl), So), Ck), dx))))));
                    else
                        X = 0.5f;
                }
                // D = (Km*(1 - X) + dt/2): integer literals, the same words
                D = __fadd_rn(__fmul_rn(Km, __fsub_rn(1.0f, X)), __fdiv_rn(dt, 2.0f));
                if (D == 0.0f) return 2;
                C1 = __fdiv_rn(__fadd_rn(__fmul_rn(Km, X), __fdiv_rn(dt, 2.000000f)), D);
                C2 = __fdiv_rn(__fsub_rn(__fdiv_rn(dt, 2.000000f), __fmul_rn(Km, X)), D);
                C3 = __fdiv_rn(__fsub_rn(__fmul_rn(Km, __fsub_rn(1.000000f, X)), __fdiv_rn(dt, 2.000000f)), D);
                if (ChannK <= 0.0f) C4 = __fdiv_rn(__fmul_rn(ql, dt), D);
                else C4 = __fdiv_rn(__fmul_rn(__fsub_rn(ql, __fmul_rn(__fmul_rn(ChannK, dx), WPk)), dt), D);
                {
                    float s = __fadd_rn(__fadd_rn(__fmul_rn(C1, qup), __fmul_rn(C2, quc)), __fmul_rn(C3, qdp));
                    if ((C4 < 0.0f) && (fabsf(C4) > s)) C4 = -s;
                    if (__fadd_rn(WP, WPC) > 0.0f) {
                        if ((C4 < 0.0f) && (fabsf(C4) > s)) C4 = -s;
                        Qj = __fsub_rn(__fadd_rn(s, C4), mc_qmn(WP, WPC, n, nCC, AREA, AREAC, R, So));
                    }
                }
                if (__fsub_rn(Qj_0, Qj) != 0.0f) {
                    h_1 = __fsub_rn(h, __fdiv_rn(__fmul_rn(Qj, __fsub_rn(h_0, h)), __fsub_rn(Qj_0, Qj)));
                    if (h_1 < 0.0f) h_1 = h;
                } else {
                    h_1 = h;
                }
                if (h > 0.0f) {
                    rerror = fabsf(__fdiv_rn(__fsub_rn(h_1, h), h));
                    aerror = fabsf(__fsub_rn(h_1, h));
                } else {
                    rerror = 0.0f;
                    aerror = 0.9f;
                }
                h_0 = fmaxf(0.0f, h);
                h = fmaxf(0.0f, h_1);
                iter = iter + 1;
                if (h < mindepth) break;   // goto 111
            }
            if (iter >= maxiter) {
                tries = tries + 1;
                if (tries <= 4) {
                    h = __fmul_rn(h, 1.33f);
                    h_0 = __fmul_rn(h_0, 0.67f);
                    maxiter = maxiter + 25;
                    continue;            // goto 110
                }
                nonconv = 1;
            }
            break;
        }
        float s = __fadd_rn(__fadd_rn(__fmul_rn(C1, qup), __fmul_rn(C2, quc)), __fmul_rn(C3, qdp));
        if (__fadd_rn(s, C4) < 0.0f) {
            if ((C4 < 0.0f) && (fabsf(C4) > s)) qdc = 0.0f;
            else qdc = fmaxf(__fadd_rn(__fadd_rn(__fmul_rn(C1, qup), __fmul_rn(C2, quc)), C4),
                             __fadd_rn(__fadd_rn(__fmul_rn(C1, qup), __fmul_rn(C3, qdp)), C4));
        } else {
            qdc = __fadd_rn(s, C4);
        }
        Twl = __fadd_rn(Bw, __fmul_rn(__fmul_rn(2.0f, z), h));
        float half = __fdiv_rn(__fsub_rn(Twl, Bw), 2.0f);
        R = __fdiv_rn(__fdiv_rn(__fmul_rn(h, __fadd_rn(Bw, Twl)), 2.0f),
                      __fadd_rn(Bw, __fmul_rn(2.0f, gfk_pow(__fadd_rn(gfk_pow(half, 2.0f),
                                                                       __fmul_rn(h, h)), 0.5f))));
        vel = __fmul_rn(__fmul_rn(__fdiv_rn(1.0f, n), gfk_pow(R, 2.0f / 3.0f)), __fsqrt_rn(So));
        depth = h;
    } else {
        // WRF leaves WPk, D and C4 as they were on the stack here; the
        // port's are zero, so qloss is defined where WRF's is not.
        qdc = 0.0f;
        depth = 0.0f;
    }
    qloss = __fmul_rn(__fmul_rn(ChannK, dx), WPk);
    if (D != 0.0f && __fdiv_rn(__fmul_rn(qloss, dt), D) > __fsub_rn(__fdiv_rn(__fmul_rn(ql, dt), D), C4)) {
        qloss = __fsub_rn(ql, __fmul_rn(C4, __fdiv_rn(D, dt)));
    }
    return 0;
}

// drive_CHANNEL :716-751: lateral inflow per reach, REAL*8 over its pixels in
// pixel order; each pixel's term is formed in REAL.  QLateral = LQLateral for
// all NLINKS entries.
extern "C" __global__
void ch_lateral(const int* __restrict__ start, const int* __restrict__ pix,
                const int* __restrict__ lake, const float* __restrict__ qstrmvol,
                const float* __restrict__ qinflowbase, const float* __restrict__ lakeinflo,
                const int* __restrict__ cell, const float* __restrict__ node_area,
                float dt, float* __restrict__ qlateral, int nlinks)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nlinks) return;
    double acc = 0.0;
    for (int p = start[k]; p < start[k + 1]; ++p) {
        int jj = pix[p], c = cell[jj];
        float w = lake[p] ? lakeinflo[c] : qstrmvol[c];
        float term = __fdiv_rn(__fmul_rn(__fdiv_rn(__fadd_rn(w, qinflowbase[c]), 1000.0f),
                                         node_area[jj]), dt);
        acc = __dadd_rn(acc, (double)term);
    }
    qlateral[k] = __double2float_rn(acc);
}

// One substep of drive_CHANNEL's reach loop (:778-851), one thread per reach.
extern "C" __global__
void ch_reach(int option, const int* __restrict__ ustart, const int* __restrict__ ups,
              const float* __restrict__ qlink1, const float* __restrict__ qlink2,
              const float* __restrict__ qlateral, const int* __restrict__ typel,
              const float* __restrict__ musk, const float* __restrict__ musx,
              const float* __restrict__ so, const float* __restrict__ chanlen,
              const float* __restrict__ mannn, const float* __restrict__ chsslp,
              const float* __restrict__ bw, const float* __restrict__ tw,
              const float* __restrict__ tw_cc, const float* __restrict__ n_cc,
              const float* __restrict__ channk, float dtrt, float* __restrict__ tmp,
              float* __restrict__ velocity, float* __restrict__ qloss,
              float* __restrict__ hlink, int* __restrict__ err, int nlinksl)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nlinksl) return;
    float quc = 0.0f, qup = 0.0f;
    for (int p = ustart[k]; p < ustart[k + 1]; ++p) {
        int m = ups[p];
        quc = __fadd_rn(quc, qlink2[m]);
        qup = __fadd_rn(qup, qlink1[m]);
    }
    tmp[k] = 0.0f;
    if (typel[k] == 1) return;               // a reservoir: the reach branch leaves it
    if (option == 1) {
        tmp[k] = mc_musking(qup, __fadd_rn(quc, qlateral[k]), qlink1[k], dtrt, musk[k], musx[k]);
    } else {
        float qdc = tmp[k], v = velocity[k], ql = qloss[k], d = hlink[k];
        int nonconv = 0;
        int e = mc_cunge(qdc, v, ql, qup, quc, qlink1[k], qlateral[k], dtrt, so[k], chanlen[k],
                         mannn[k], chsslp[k], bw[k], tw[k], tw_cc[k], n_cc[k], d, channk[k], nonconv);
        if (e) atomicOr(err, e);
        if (nonconv) atomicOr(err, 4);
        tmp[k] = qdc;
        velocity[k] = v;
        qloss[k] = ql;
        hlink[k] = d;
    }
}

extern "C" __global__
void ch_commit(const int* __restrict__ typel, const float* __restrict__ tmp,
               float* __restrict__ qlink1, float* __restrict__ qlink2, int nlinksl)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nlinksl) return;
    if (typel[k] != 1) qlink2[k] = tmp[k];
    qlink1[k] = qlink2[k];
}
