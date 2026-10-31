// woof/core/kernels/sms3dtke.cu
//
// WRF v4.7.1 km_opt = 5: the SMS-3DTKE scale-adaptive 3-D TKE closure
// (Zhang, Wang et al. 2018, MWR), transcribed from
// dyn_em/module_diffusion_em.F (sha256 a7d4570c...) and called in the order
// dyn_em/module_first_rk_step_part2.F gives it (:582-660, :904-933,
// :1028-1052):
//
//   nonlocal_flux      (:4911-5239)  prescribed nonlocal heat flux NLFLUX,
//                                    the nonlocal momentum gammas GAMU/GAMV,
//                                    the theta-method PBLH/KPBL
//   free_atmos_length  (:2665-2860)  BouLac free-atmosphere length ELMIN
//   meso_length_scale  (:2411-2662)  Nakanishi-Niino mesoscale length DLK
//   tke_km, km_opt = 5 (:2253-2333)  the blended horizontal/vertical K and
//                                    the mesoscale-limit XKMV_MESO
//   tke_rhs, km_opt=5  (:6099-6229)  buoyancy with the nonlocal flux, the
//                                    dissipation length L_DISS, limiter
//   vertical_diffusion_implicit (:7616-8475, tridiag :7583-7613) the
//                                    implicit vertical mixing of u, v, w,
//                                    theta, TKE (with the implicit
//                                    dissipation) and every scalar
//
// and the partial functions pthnl/pthl/pu (:5245-5326).  Every routine is a
// statement-for-statement transcription: Fortran's left-to-right operand
// order, REAL**REAL as glibc powf (gfk_pow, also for the literal **2. and
// **b1 = 2.0, which gfortran -O0 lowers to a powf call), INTEGER powers as
// libgcc's __powisf2 (x**3 = x*(x*x), x**4 = (x*x)*(x*x)), EXP as glibc expf
// (gfk_exp) and TANH as glibc 2.43's correctly rounded tanhf (sfire_tanhf).
// The unit compiles without contraction and without flush-to-zero
// (woof/core/kernels/__init__.py), as gfortran -O0 -ffp-contract=off does.
// Graded word for word against the compiled WRF routines by
// tests/test_sms3dtke_wrf471_parity.py (tools/sms3dtke_wrf471_oracle).
//
// Storage.  woof's C order: mass (nz, ny, nx), w levels (nz + 1, ny, nx),
// u (nz, ny, nx + 1), v (nz, ny + 1, nx), planes (ny, nx).  A Fortran level
// k (kts = 1 .. kde = nz + 1) is C index k - 1; the macros below take the
// FORTRAN k so each statement reads as the source does.  i and j are C
// (0-based) indices.  bx/by select WRF's lateral bounds: 0 is periodic (the
// stencil wraps, as WRF's periodic halo does), 1 is specified/nested (the
// open_x* / specified / nested trimming: ids+1 .. ide-2 for mass points).
// Radiative open boundaries are refused at the config door (km_opt = 5's
// TKE gets no set_physical_bc3d in WRF, solve_em.F:3332).

#define SMS_G       9.81f
#define SMS_RD      287.0f
#define SMS_CP      (7.0f * SMS_RD / 2.0f)
#define SMS_RV      461.6f
#define SMS_RCP     (SMS_RD / SMS_CP)
#define SMS_KARMAN  0.4f
#define SMS_P1000   100000.0f
#define SMS_EPS15   1.0e-15f
#define SMS_PRANDTL (1.0f / 3.0f)
#define SMS_RVOVRD  (SMS_RV / SMS_RD)

// Fortran-k accessors.  M: mass (nz levels), W: w levels (nz + 1).
#define SMS_M(a, k, j, i)  (a)[((size_t)((k) - 1) * ny + (size_t)(j)) * nx + (size_t)(i)]
#define SMS_W(a, k, j, i)  (a)[((size_t)((k) - 1) * ny + (size_t)(j)) * nx + (size_t)(i)]
#define SMS_U(a, k, j, i)  (a)[((size_t)((k) - 1) * ny + (size_t)(j)) * (nx + 1) + (size_t)(i)]
#define SMS_V(a, k, j, i)  (a)[((size_t)((k) - 1) * (ny + 1) + (size_t)(j)) * nx + (size_t)(i)]
#define SMS_P(a, j, i)     (a)[(size_t)(j) * nx + (size_t)(i)]

// Lateral index of a mass-point neighbour: the periodic halo is the wrap.
__device__ __forceinline__ int sms_wx(int i, int nx, int bx)
{
    return bx ? i : PERIODIC(i, nx);
}

// WRF's mass-point tile bounds (its..min(ite,ide-1), trimmed by one row at
// an open/specified/nested side).
__device__ __forceinline__ bool sms_mass_in(int i, int j, int nx, int ny,
                                            int bx, int by)
{
    if (i < (bx ? 1 : 0) || i > (bx ? nx - 2 : nx - 1)) return false;
    if (j < (by ? 1 : 0) || j > (by ? ny - 2 : ny - 1)) return false;
    return true;
}

// x**3 and x**4 with an INTEGER exponent: gfortran -O0 calls libgcc's
// __powisf2, whose square-and-multiply loop rounds after every product.
__device__ __forceinline__ float sms_powi3(float x)
{
    float y = x;
    float x2 = x * x;
    return y * x2;
}

__device__ __forceinline__ float sms_powi4(float x)
{
    float x2 = x * x;
    return x2 * x2;
}

__device__ __forceinline__ float sms_powi2(float x)
{
    return x * x;
}

// ---------------------------------------------------------------------------
// The partial functions (:5245-5326).  b1 = 2.0 is a REAL parameter, so
// (doh)**b1 is a powf call; the d <= 100 m LES cut comes last.
__device__ float sms_pthnl(float d, float h)
{
    const float a1 = 1.000f, a2 = 0.936f, a3 = -1.110f,
                a4 = 1.000f, a5 = 0.312f, a6 = 0.329f, a7 = 0.243f;
    const float b1 = 2.0f, b2 = 0.875f;
    float p;
    if (h != 0.0f) {
        float doh = d / h;
        float num = a1 * gfk_pow(doh, b1) + a2 * gfk_pow(doh, b2) + a3;
        float den = a4 * gfk_pow(doh, b1) + a5 * gfk_pow(doh, b2) + a6;
        p = a7 * num / den + (1.0f - a7);
    } else {
        p = 1.0f;
    }
    p = fmaxf(p, 0.0f);
    p = fminf(p, 1.0f);
    if (d <= 100.0f) p = 0.0f;
    return p;
}

__device__ float sms_pthl(float d, float h)
{
    const float a1 = 1.000f, a2 = 0.870f, a3 = -0.913f,
                a4 = 1.000f, a5 = 0.153f, a6 = 0.278f, a7 = 0.280f;
    const float b1 = 2.0f, b2 = 0.5f;
    float p;
    if (h != 0.0f) {
        float doh = d / h;
        float num = a1 * gfk_pow(doh, b1) + a2 * gfk_pow(doh, b2) + a3;
        float den = a4 * gfk_pow(doh, b1) + a5 * gfk_pow(doh, b2) + a6;
        p = a7 * num / den + (1.0f - a7);
    } else {
        p = 1.0f;
    }
    p = fmaxf(p, 0.0f);
    p = fminf(p, 1.0f);
    if (d <= 100.0f) p = 0.0f;
    return p;
}

__device__ float sms_pu(float d, float h)
{
    const float a1 = 1.0f, a2 = 0.070f, a3 = 1.0f, a4 = 0.142f, a5 = 0.071f;
    const float b1 = 2.0f, b2 = 0.6666667f;
    float p;
    if (h != 0.0f) {
        float doh = d / h;
        float num = a1 * gfk_pow(doh, b1) + a2 * gfk_pow(doh, b2);
        float den = a3 * gfk_pow(doh, b1) + a4 * gfk_pow(doh, b2) + a5;
        p = num / den;
    } else {
        p = 1.0f;
    }
    p = fmaxf(p, 0.0f);
    p = fminf(p, 1.0f);
    if (d <= 100.0f) p = 0.0f;
    return p;
}

// SQRT(dx/msftx*dy/msfty): ((dx/m)*dy)/m, left to right.
__device__ __forceinline__ float sms_delxy(float dx, float dy, float m)
{
    return sqrtf(dx / m * dy / m);
}

// dthrdn, the centred d(theta)/dz with WRF's one-sided surface and top
// forms (tke_km :2181-2201, meso_length_scale :2519-2541).  p8w/t8w are the
// phy_prep planes at kts and kde.
__device__ float sms_dthrdn(const float* th, const float* rdz,
                            const float* rdzw,
                            const float* p8w1, const float* t8w1,
                            const float* p8wt, const float* t8wt,
                            int k, int ktf, int j, int i, int ny, int nx)
{
    float tmpdz;
    if (k == 1) {
        tmpdz = 1.0f / SMS_M(rdzw, k + 1, j, i) + 1.0f / SMS_M(rdzw, k, j, i);
        float thetasfc = SMS_P(t8w1, j, i)
                       / gfk_pow(SMS_P(p8w1, j, i) / SMS_P1000, SMS_RCP);
        return (SMS_M(th, k + 1, j, i) - thetasfc) / tmpdz;
    }
    if (k == ktf) {
        tmpdz = 1.0f / SMS_M(rdz, k, j, i) + 0.5f / SMS_M(rdzw, k, j, i);
        float thetatop = SMS_P(t8wt, j, i)
                       / gfk_pow(SMS_P(p8wt, j, i) / SMS_P1000, SMS_RCP);
        return (thetatop - SMS_M(th, k - 1, j, i)) / tmpdz;
    }
    tmpdz = 1.0f / SMS_M(rdz, k + 1, j, i) + 1.0f / SMS_M(rdz, k, j, i);
    return (SMS_M(th, k + 1, j, i) - SMS_M(th, k - 1, j, i)) / tmpdz;
}

// ---------------------------------------------------------------------------
// nonlocal_flux (:4911-5239), one thread per column.  Writes NLFLUX (w-level
// storage, levels kts..ktf; kde is never written, as in WRF), PBLH and KPBL
// for every column in bounds, GAMU/GAMV only where the column is convective
// (INTENT(OUT) arrays WRF leaves holding the previous step's value
// elsewhere -- the reason Registry gives them the restart flag), and with
// isfflx 0 or 2 the prescribed sensible heat flux into HFX.
extern "C" __global__
void sms_nonlocal_flux(const float* rdzw, const float* th, const float* qv,
                       int has_qv,
                       const float* u_phy, const float* v_phy,
                       const float* rho,
                       const float* ust, const float* br, const float* wspd,
                       const float* msft, float* hfx,
                       float dx, float dy, int isfflx, float tke_heat_flux,
                       float* nlflux, float* gamu, float* gamv,
                       float* hpbl, int* kpbl,
                       int nz, int ny, int nx, int bx, int by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny || !sms_mass_in(i, j, nx, ny, bx, by)) return;

    const float phifac = 8.0f, d1 = 0.02f, d2 = 0.05f, zfmin = 1.e-8f;
    const float h1 = 0.33333335f, h2 = 0.6666667f, tmin = 1.e-2f;
    const float mltop = 1.0f, sfcfracn1 = 0.075f;
    const float nlfrac = 0.68f, enlfrac = -0.15f;
    const float a11 = 1.0f, a12 = -1.15f, ezfac = 1.5f;
    const float cpent = -0.4f, rigsmax = 100.0f, rimin = -100.0f;
    const float entfmin = 1.0f, entfmax = 5.0f, sm = 10.9f;
    const float ep1 = SMS_RV / SMS_RD - 1.0f;
    const int ktf = nz;

#define QV1(k) (has_qv ? SMS_M(qv, k, j, i) : 0.0f)
#define THV(k) (SMS_M(th, k, j, i) * (1.0f + ep1 * QV1(k)))

    // zq(1) = 0; zq(k+1) = 1/rdzw(k) + zq(k) -- recomputed as a running
    // sum wherever it is needed, in the same order.
    for (int k = 1; k <= ktf; ++k) SMS_W(nlflux, k, j, i) = 0.0f;

    float govrth = SMS_G / SMS_M(th, 1, j, i);

    float cpm, heat_flux;
    if (isfflx == 0 || isfflx == 2) {
        heat_flux = tke_heat_flux;
        cpm = SMS_CP * (1.0f + 0.8f * QV1(1));
        SMS_P(hfx, j, i) = heat_flux * cpm * SMS_M(rho, 1, j, i);
    }

    int kp = 1;
    float hp = 0.0f;            // zq(i,1,j)
    bool pblflg = true, sfcflg = true;
    cpm = SMS_CP * (1.0f + 0.8f * QV1(1));
    float sflux = (SMS_P(hfx, j, i) / cpm) / SMS_M(rho, 1, j, i);
    if (SMS_P(br, j, i) > 0.0f) sfcflg = false;

    // PBL height, theta method (:5122-5134).
    {
        float thsfc = THV(1) + 0.5f;
        float zq_km1 = 0.0f;                      // zq(k-1)
        float zq_k = 1.0f / SMS_M(rdzw, 1, j, i); // zq(k)   (k = 2)
        for (int k = 2; k <= ktf; ++k) {
            float zq_kp1 = 1.0f / SMS_M(rdzw, k, j, i) + zq_k;
            if (THV(k) > thsfc) {
                float za_km1 = 0.5f * (zq_km1 + zq_k);
                float za_k = 0.5f * (zq_k + zq_kp1);
                hp = za_km1 + (thsfc - THV(k - 1))
                   / (fmaxf(0.01f, THV(k) - THV(k - 1))) * (za_k - za_km1);
                kp = k;
                break;
            }
            zq_km1 = zq_k;
            zq_k = zq_kp1;
        }
    }
    float zq2 = 1.0f / SMS_M(rdzw, 1, j, i);      // zq(i,2,j)
    if (hp < zq2) kp = 1;
    if (kp <= 1) pblflg = false;

    float wstar3;
    if (sfcflg) {
        float bfx0 = fmaxf(sflux, 0.0f);
        wstar3 = govrth * bfx0 * hp;             // wstar = wstar3**h1 is unused
    } else {
        wstar3 = 0.0f;
    }
    float us = SMS_P(ust, j, i);
    float ust3 = sms_powi3(us);
    float wscale = gfk_pow(ust3 + phifac * SMS_KARMAN * wstar3 * 0.5f, h1);
    wscale = fminf(wscale, us * 16.0f);
    wscale = fmaxf(wscale, us / 5.0f);

    float m = SMS_P(msft, j, i);
    float delxy = sms_delxy(dx, dy, m);
    float pu1 = sms_pu(delxy, hp);
    if ((sfcflg && sflux > 0.0f) && (hp > 0.0f)) {
        float brint = -sm * us * us * wstar3 / (hp * sms_powi4(wscale));
        SMS_P(gamu, j, i) = pu1 * brint * SMS_M(u_phy, 1, j, i)
                          / SMS_P(wspd, j, i);
        SMS_P(gamv, j, i) = pu1 * brint * SMS_M(v_phy, 1, j, i)
                          / SMS_P(wspd, j, i);
    } else {
        pblflg = false;
    }

    SMS_P(hpbl, j, i) = hp;
    SMS_P(kpbl, j, i) = kp;
    if (!pblflg) return;

    // :5160-5190
    float deltaoh, enlfrac2, hfxpbl;
    {
        int k = kp - 1;
        // za(k): zq(k) and zq(k+1).
        float zqk = 0.0f;
        for (int kk = 1; kk < k; ++kk) zqk = 1.0f / SMS_M(rdzw, kk, j, i) + zqk;
        float zqk1 = 1.0f / SMS_M(rdzw, k, j, i) + zqk;
        float za_k = 0.5f * (zqk + zqk1);
        float wm3 = wstar3 + 5.0f * sms_powi3(us);
        float wm2 = gfk_pow(wm3, h2);
        float bfxpbl = -0.15f * THV(1) / SMS_G * wm3 / hp;
        float dthv = fmaxf(THV(k + 1) - THV(k), tmin);
        float dtheta = fmaxf(SMS_M(th, k + 1, j, i) - SMS_M(th, k, j, i), tmin);
        float du = SMS_M(u_phy, k + 1, j, i) - SMS_M(u_phy, k, j, i);
        float dv = SMS_M(v_phy, k + 1, j, i) - SMS_M(v_phy, k, j, i);
        float we = fmaxf(bfxpbl / dthv, -sqrtf(wm2));
        hfxpbl = we * dtheta;
        float delb = govrth * dthv;
        deltaoh = d1 * hp + d2 * wm2 / delb;
        deltaoh = fmaxf(ezfac * deltaoh, hp - za_k - 1.0f);
        deltaoh = fminf(deltaoh, hp);
        float rigs;
        if ((du != 0.0f) || (dv != 0.0f)) {
            rigs = govrth * dthv * deltaoh
                 / (gfk_pow(du, 2.0f) + gfk_pow(dv, 2.0f));
            rigs = fmaxf(fminf(rigs, rigsmax), rimin);
        } else {
            rigs = rigsmax;
        }
        enlfrac2 = fmaxf(fminf(wm3 / wstar3 / (1.0f + cpent / rigs), entfmax),
                         entfmin);
        enlfrac2 = enlfrac2 * enlfrac;
    }

    // :5208-5236 (entfacmf is recomputed per level from zq(k+1)).
    float dlt = deltaoh;                           // pre-normalisation
    deltaoh = deltaoh / hp;
    float mlfrac = mltop - deltaoh;
    float zfac1 = fminf(fmaxf((zq2 / hp), zfmin), 1.0f);
    float sfcfracn = fmaxf(sfcfracn1, zfac1);
    float sflux0 = (a11 + a12 * sfcfracn) * sflux;
    float snlflux0 = nlfrac * sflux0;
    float amf1 = snlflux0 / sfcfracn;
    float amf2 = -snlflux0 / (mlfrac - sfcfracn);
    float bmf2 = -mlfrac * amf2;
    float amf3 = snlflux0 * enlfrac2 / deltaoh;
    float bmf3 = -amf3 * mlfrac;
    hfxpbl = amf3 + bmf3;
    float pth1 = sms_pthnl(delxy, hp);
    hfxpbl = hfxpbl * pth1;

    float zq_k = 0.0f;
    for (int k = 1; k <= ktf; ++k) {
        float zq_kp1 = 1.0f / SMS_M(rdzw, k, j, i) + zq_k;
        float zfacmf = fmaxf((zq_kp1 / hp), zfmin);
        if (k < kp) {
            float e = (zq_kp1 - hp) / dlt;
            float entfacmf = sqrtf(gfk_pow(e, 2.0f));
            float v = SMS_W(nlflux, k, j, i);
            if (zfacmf <= sfcfracn) {
                v = amf1 * zfacmf;
            } else if (zfacmf <= mlfrac) {
                v = amf2 * zfacmf + bmf2;
            }
            v = v + hfxpbl * gfk_exp(-entfacmf);
            v = v * pth1;
            SMS_W(nlflux, k, j, i) = v;
        }
        zq_k = zq_kp1;
    }
#undef QV1
#undef THV
}

// ---------------------------------------------------------------------------
// free_atmos_length (:2665-2860), one thread per column: BouLac's upward and
// downward parcel displacement lengths, ELMIN = min(dlu, dld)/(1+min/2000).
extern "C" __global__
void sms_free_atmos_length(const float* rdzw, const float* tke,
                           const float* th, float* elmin,
                           int nz, int ny, int nx, int bx, int by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny || !sms_mass_in(i, j, nx, ny, bx, by)) return;
    const int kts = 1, ktf = nz;
#define TH(k) SMS_M(th, k, j, i)
#define DZW(k) (1.0f / SMS_M(rdzw, k, j, i))
    float ztop = 0.0f;                              // zfull(ktf+1)
    for (int k = kts; k <= ktf; ++k) ztop = DZW(k) + ztop;
    float zk = 0.0f;                                // zfull(k)
    for (int k = kts; k <= ktf; ++k) {
        float zk1 = DZW(k) + zk;                    // zfull(k+1)
        float tk = SMS_M(tke, k, j, i);
        float zup = 0.0f;
        float dlu = ztop - zk - 1.0f / SMS_M(rdzw, k, j, i) / 2.0f;
        float zzz = 0.0f;
        float zup_inf = 0.0f;
        float beta = SMS_G / 300.0f;
        if (k < ktf) {
            int izz = k;
            for (;;) {
                if (izz < ktf) {
                    float dzt = (DZW(izz + 1) + DZW(izz)) / 2.0f;
                    zup = zup - beta * TH(k) * dzt;
                    zup = zup + beta * (TH(izz + 1) + TH(izz)) * dzt / 2.0f;
                    zzz = zzz + dzt;
                    if (tk < zup && tk >= zup_inf) {
                        float bbb = (TH(izz + 1) - TH(izz)) / dzt;
                        float tl;
                        if (bbb != 0.0f) {
                            float bt = beta * (TH(izz) - TH(k));
                            tl = (-(beta * (TH(izz) - TH(k)))
                                  + sqrtf(fmaxf(0.0f, gfk_pow(bt, 2.0f)
                                        + 2.0f * bbb * beta * (tk - zup_inf))))
                                 / bbb / beta;
                        } else {
                            if (TH(izz) != TH(k)) {
                                tl = (tk - zup_inf) / (beta * (TH(izz) - TH(k)));
                            } else {
                                tl = 0.0f;
                            }
                        }
                        dlu = zzz - dzt + tl;
                        break;
                    }
                    zup_inf = zup;
                    izz = izz + 1;
                } else {
                    break;
                }
            }
        }

        float zdo = 0.0f, zdo_sup = 0.0f;
        float dld = zk + 1.0f / SMS_M(rdzw, k, j, i) / 2.0f;
        zzz = 0.0f;
        if (k > kts) {
            int izz = k;
            for (;;) {
                if (izz > kts) {
                    float dzt = (DZW(izz - 1) + DZW(izz)) / 2.0f;
                    zdo = zdo + beta * TH(k) * dzt;
                    zdo = zdo - beta * (TH(izz - 1) + TH(izz)) * dzt / 2.0f;
                    zzz = zzz + dzt;
                    if (tk < zdo && tk >= zdo_sup) {
                        float bbb = (TH(izz) - TH(izz - 1)) / dzt;
                        float tl;
                        if (bbb != 0.0f) {
                            float bt = beta * (TH(izz) - TH(k));
                            tl = (beta * (TH(izz) - TH(k))
                                  + sqrtf(fmaxf(0.0f, gfk_pow(bt, 2.0f)
                                        + 2.0f * bbb * beta * (tk - zdo_sup))))
                                 / bbb / beta;
                        } else {
                            if (TH(izz) != TH(k)) {
                                tl = (tk - zdo_sup) / (beta * (TH(izz) - TH(k)));
                            } else {
                                tl = 0.0f;
                            }
                        }
                        dld = zzz - dzt + tl;
                        break;
                    }
                    zdo_sup = zdo;
                    izz = izz - 1;
                } else {
                    break;
                }
            }
        }
        float dlg = (zk + zk1) / 2.0f;
        dld = fminf(dld, dlg);
        float e = fminf(dlu, dld);
        e = e / (1.0f + (e / 2000.0f));
        SMS_M(elmin, k, j, i) = e;
        zk = zk1;
    }
#undef TH
#undef DZW
}

// ---------------------------------------------------------------------------
// meso_length_scale (:2411-2662), one thread per column: DLK, the
// Nakanishi-Niino length blended into 0.4*ELMIN above PBLH + the transition
// layer.  With isfflx 0 or 2 it re-writes HFX from tke_heat_flux exactly as
// nonlocal_flux did (the same word).
extern "C" __global__
void sms_meso_length_scale(const float* rdzw, const float* rdz,
                           const float* tke, const float* th,
                           const float* p8w1, const float* t8w1,
                           const float* p8wt, const float* t8wt,
                           const float* hpbl, const float* elmin,
                           const float* rmol, const float* rho,
                           float* hfx, const float* qv, int has_qv,
                           int isfflx, float tke_heat_flux,
                           float* dlk,
                           int nz, int ny, int nx, int bx, int by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny || !sms_mass_in(i, j, nx, ny, bx, by)) return;
    const int kts = 1, ktf = nz;
    const float alp1 = 0.8f, alp2 = 1.0f, alp3 = 1.0f, alp4 = 100.0f;
    const float minzi = 300.0f, maxdz = 750.0f, mindz = 300.0f;

    float qv1 = has_qv ? SMS_M(qv, 1, j, i) : 0.0f;
    // elt and vsc: sum over k of qtke*dz*za and qtke*dz.
    float elt = 1.0e-5f, vsc = 1.0e-5f;
    {
        float zk = 0.0f;
        for (int k = kts; k <= ktf; ++k) {
            float zk1 = 1.0f / SMS_M(rdzw, k, j, i) + zk;
            float za = (zk + zk1) / 2.0f;
            float qtke = sqrtf(2.0f * SMS_M(tke, k, j, i));
            float qdz = qtke * (1.0f / SMS_M(rdzw, k, j, i));
            elt = elt + qdz * za;
            vsc = vsc + qdz;
            zk = zk1;
        }
    }
    elt = alp1 * elt / vsc;

    if (isfflx == 0 || isfflx == 2) {
        float cpm = SMS_CP * (1.0f + 0.8f * qv1);
        SMS_P(hfx, j, i) = tke_heat_flux * cpm * SMS_M(rho, 1, j, i);
    }
    float cpm = SMS_CP * (1.0f + 0.8f * qv1);
    float sflux = (SMS_P(hfx, j, i) / cpm) / SMS_M(rho, 1, j, i);

    float gtr = SMS_G / 300.0f;
    float rm = SMS_P(rmol, j, i);
    float hp = SMS_P(hpbl, j, i);
    float zk = 0.0f;
    for (int k = kts; k <= ktf; ++k) {
        float zk1 = 1.0f / SMS_M(rdzw, k, j, i) + zk;
        float za = (zk + zk1) / 2.0f;
        float qtke = sqrtf(2.0f * SMS_M(tke, k, j, i));
        float dthrdn = sms_dthrdn(th, rdz, rdzw, p8w1, t8w1, p8wt, t8wt,
                                  k, ktf, j, i, ny, nx);
        float elb, elf;
        if (dthrdn > 0.0f) {
            float N2 = gtr * dthrdn;
            float qcv = gfk_pow(gtr * fmaxf(sflux, 0.0f) * elt, 1.0f / 3.0f);
            elb = qtke / sqrtf(N2)
                * (alp2 + alp3 * sqrtf(qcv / (elt * sqrtf(N2))));
            elf = alp2 * qtke / sqrtf(N2);
        } else {
            elb = 1.0e10f;
            elf = elb;
        }
        float els;
        if (rm > 0.0f) {
            els = SMS_KARMAN * za / (1.0f + 2.7f * fminf(za * rm, 1.0f));
        } else {
            float coe = gfk_pow(1.0f - alp4 * za * rm, 0.2f);
            els = 1.0f * SMS_KARMAN * za * coe;
        }
        float d = fminf(elb / (elb / elt + elb / els + 1.0f), elf);
        float zi2 = fmaxf(hp, minzi);
        float hh1 = fmaxf(0.3f * zi2, mindz);
        hh1 = fminf(hh1, maxdz);
        float hh2 = hh1 / 2.0f;
        float wt = 0.5f * sfire_tanhf((za - (zi2 + hh1)) / hh2) + 0.5f;
        d = d * (1.0f - wt) + 0.4f * SMS_M(elmin, k, j, i) * wt;
        SMS_M(dlk, k, j, i) = d;
        zk = zk1;
    }
}

// ---------------------------------------------------------------------------
// tke_km's km_opt = 5 branch (:2253-2333) with its dthrdn (:2181-2201): the
// deformation (smag_2d) horizontal K blended with the TKE horizontal K by
// pthl, and the LES and mesoscale vertical K blended by pu.  Points outside
// WRF's bounds are written 0, the never-written cold state.
extern "C" __global__
void sms_tke_km(const float* tke, const float* th,
                const float* rdz, const float* rdzw,
                const float* p8w1, const float* t8w1,
                const float* p8wt, const float* t8wt,
                const float* d11, const float* d22, const float* d12,
                const float* zx, const float* zy, const float* msft,
                const float* hpbl, const float* dlk,
                float dx, float dy, float c_s, float c_k, float tke_seed,
                float* xkmh, float* xkmv, float* xkhh, float* xkhv,
                float* xkmv_meso,
                int nz, int ny, int nx, int bx, int by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int kc = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || kc >= nz) return;
    int k = kc + 1;
    if (!sms_mass_in(i, j, nx, ny, bx, by)) {
        SMS_M(xkmh, k, j, i) = 0.0f;
        SMS_M(xkmv, k, j, i) = 0.0f;
        SMS_M(xkhh, k, j, i) = 0.0f;
        SMS_M(xkhv, k, j, i) = 0.0f;
        SMS_M(xkmv_meso, k, j, i) = 0.0f;
        return;
    }
    const int ktf = nz;
    int ip = sms_wx(i + 1, nx, bx);
    int jp = by ? j + 1 : PERIODIC(j + 1, ny);
    float m = SMS_P(msft, j, i);
    float pr = SMS_PRANDTL;

    // def2 (:2262-2271)
    float a = SMS_M(d11, k, j, i), b = SMS_M(d22, k, j, i);
    float def2 = 0.25f * ((a - b) * (a - b));
    float tmp = 0.25f * (SMS_M(d12, k, j, i) + SMS_M(d12, k, jp, i)
                         + SMS_M(d12, k, j, ip) + SMS_M(d12, k, jp, ip));
    def2 = def2 + tmp * tmp;

    // smag_2d horizontal K (:2273-2297).  zx is the u-face slope on w
    // levels: face i (west) and i+1 (east); zy the v-face slope.
    float mlen_h = sqrtf(dx / m * dy / m);
    tmp = sqrtf(def2);
    float xkmh_s = c_s * c_s * mlen_h * mlen_h * tmp;
    xkmh_s = fminf(xkmh_s, 10.0f * mlen_h);
    float xkhh_s = xkmh_s / pr;
    {
        float dxm = dx / m;
        float dym = dy / m;
#define ZX(kk, ii) zx[((size_t)((kk) - 1) * ny + (size_t)(j)) * (nx + 1) + (size_t)(ii)]
#define ZY(kk, jj) zy[((size_t)((kk) - 1) * (ny + 1) + (size_t)(jj)) * nx + (size_t)(i)]
        float rw = SMS_M(rdzw, k, j, i);
        float tmpzx = (0.25f * (fabsf(ZX(k, i)) + fabsf(ZX(k, i + 1))
                                + fabsf(ZX(k + 1, i)) + fabsf(ZX(k + 1, i + 1)))
                       * rw * dxm);
        float tmpzy = (0.25f * (fabsf(ZY(k, j)) + fabsf(ZY(k, j + 1))
                                + fabsf(ZY(k + 1, j)) + fabsf(ZY(k + 1, j + 1)))
                       * rw * dym);
#undef ZX
#undef ZY
        float alpha = fmaxf(sqrtf(tmpzx * tmpzx + tmpzy * tmpzy), 1.0f);
        float def_limit = fmaxf(10.0f / mlen_h, 1.e-3f);
        if (tmp > def_limit) {
            xkmh_s = xkmh_s / (alpha * alpha);
        } else {
            xkmh_s = xkmh_s / (alpha);
        }
        xkhh_s = xkmh_s / pr;
    }

    // TKE K and the blend (:2300-2329).
    float dthrdn = sms_dthrdn(th, rdz, rdzw, p8w1, t8w1, p8wt, t8wt,
                              k, ktf, j, i, ny, nx);
    tmp = sqrtf(fmaxf(SMS_M(tke, k, j, i), tke_seed));
    float deltas = 1.0f / SMS_M(rdzw, k, j, i);
    float mlen_v = deltas;
    if (dthrdn > 0.0f) {
        float mlen_s = 0.76f * tmp
                     / gfk_pow(fabsf(SMS_G / SMS_M(th, k, j, i) * dthrdn), 0.5f);
        mlen_v = fminf(mlen_v, mlen_s);
    }
    float delxy = sqrtf(dx / m * dy / m);
    float hp = SMS_P(hpbl, j, i);
    float pth1 = sms_pthl(delxy, hp);
    float pu1 = sms_pu(delxy, hp);

    const float xkmvo = 0.0f, xkhvo = 0.0f;
    float xkmh_t = c_k * tmp * mlen_h;
    float meso = 0.4f * tmp * SMS_M(dlk, k, j, i);
    meso = meso + xkmvo;
    float xkmv_les = c_k * tmp * mlen_v;
    float kmv = (1.0f - pu1) * xkmv_les + pu1 * meso;
    kmv = fminf(kmv, 1000.0f);

    float pr_inv_h = 1.0f / SMS_PRANDTL;
    float pr_inv_v_les = 1.0f + 2.0f * mlen_v / deltas;
    float pr_inv_v_meso = 1.0f;
    float xkhh_t = xkmh_t * pr_inv_h;
    float khv = kmv * (1.0f - pth1) * pr_inv_v_les
              + pth1 * (pr_inv_v_meso * kmv + xkhvo);
    khv = fminf(khv, 1000.0f);
    SMS_M(xkmh, k, j, i) = pth1 * xkmh_s + (1.0f - pth1) * xkmh_t;
    SMS_M(xkhh, k, j, i) = pth1 * xkhh_s + (1.0f - pth1) * xkhh_t;
    SMS_M(xkmv, k, j, i) = kmv;
    SMS_M(xkhv, k, j, i) = khv;
    SMS_M(xkmv_meso, k, j, i) = meso;
}

// ---------------------------------------------------------------------------
// tke_rhs for km_opt = 5 after tke_shear (:6177-6227): tke_buoyancy's
// km_opt = 5 arm (:6315-6321, the nonlocal heat flux enters the buoyancy
// production) and its surface level (:6340-6366), then tke_dissip's
// km_opt = 5 arm, which writes L_DISS and adds NO dissipation to the
// tendency (:6512-6520; vertical_diffusion_implicit treats it), then the
// positivity limiter.  ``tend`` holds tke_shear's tendency on entry
// (woof's wrf_tke_rhs shear term) and the complete tke_rhs on exit.
extern "C" __global__
void sms_tke_rhs(float* tend, const float* bn2, const float* xkhv,
                 const float* nlflux, const float* th, const float* tke,
                 const float* mut, const float* c1h, const float* c2h,
                 const float* rdzw, const float* msft,
                 const float* hpbl, const float* dlk,
                 const float* qv, int has_qv, const float* rho,
                 const float* hfx, int isfflx, float tke_heat_flux,
                 float c_k, float dt, float dx, float dy,
                 float* l_diss,
                 int nz, int ny, int nx, int bx, int by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int kc = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || kc >= nz) return;
    if (!sms_mass_in(i, j, nx, ny, bx, by)) return;
    int k = kc + 1;
    const int kts = 1, ktf = nz;
    float mu = SMS_P(mut, j, i);
    float c1 = c1h[k - 1], c2 = c2h[k - 1];
    float t = SMS_M(tend, k, j, i);

    // tke_buoyancy
    if (k >= kts + 1) {
        t = t - (c1 * mu + c2) * (SMS_M(xkhv, k, j, i) * SMS_M(bn2, k, j, i)
              - 0.5f * (SMS_W(nlflux, k + 1, j, i) + SMS_W(nlflux, k, j, i))
                * SMS_G / SMS_M(th, k, j, i));
    } else {
        float heat_flux;
        if (isfflx == 0 || isfflx == 2) {
            heat_flux = tke_heat_flux;
        } else {
            float q = has_qv ? SMS_M(qv, k, j, i) : 0.0f;
            float cpm = SMS_CP * (1.0f + 0.8f * q);
            heat_flux = (SMS_P(hfx, j, i) / cpm) / SMS_M(rho, k, j, i);
        }
        t = t - (c1 * mu + c2) * ((SMS_M(xkhv, k, j, i) * SMS_M(bn2, k, j, i))
              - (SMS_G / SMS_M(th, k, j, i)) * heat_flux) / 2.0f;
    }

    // tke_dissip, km_opt = 5: L_DISS only.
    {
        float m = SMS_P(msft, j, i);
        float ce1 = (c_k / 0.10f) * 0.19f;
        float ce2 = fmaxf(0.0f, 0.93f - ce1);
        float rw = SMS_M(rdzw, k, j, i);
        // calc_l_scale (:2341-2406)
        float deltas = gfk_pow(dx / m * dy / m / rw, 0.33333333f);
        float ls = deltas;
        float bn = SMS_M(bn2, k, j, i);
        float tk = SMS_M(tke, k, j, i);
        if (bn > 1.0e-6f) {
            float tmp = sqrtf(fmaxf(tk, 1.0e-6f));
            ls = 0.76f * tmp / sqrtf(bn);
            ls = fminf(ls, deltas);
            ls = fmaxf(ls, 0.001f * deltas);
        }
        // tke_dissip's own deltas (:6476), the same expression.
        deltas = gfk_pow(dx / m * dy / m / rw, 0.33333333f);
        float coefc;
        if (k == kts || k == ktf) {
            coefc = 3.9f;
        } else {
            coefc = ce1 + ce2 * ls / deltas;
        }
        float delxy = sqrtf(dx / m * dy / m);
        float pu1 = sms_pu(delxy, SMS_P(hpbl, j, i));
        float coefc_m = 0.3f;
        SMS_M(l_diss, k, j, i) = (1.0f - pu1) * ls / coefc
                               + pu1 * SMS_M(dlk, k, j, i) / coefc_m;
    }

    // tke_rhs's lower limit (:6221-6227)
    t = fmaxf(t, -(c1 * mu + c2) * fmaxf(0.0f, SMS_M(tke, k, j, i)) / dt);
    SMS_M(tend, k, j, i) = t;
}

// ---------------------------------------------------------------------------
// tridiag (:7583-7613) on a column workspace: a, b, c, d, q with stride
// ``s`` between levels.  c(n) = 0 and d becomes the solution.
__device__ void sms_tridiag(int n, float* a, float* b, float* c, float* d,
                            float* q, size_t s)
{
    c[(size_t)(n - 1) * s] = 0.0f;
    q[0] = -c[0] / b[0];
    d[0] = d[0] / b[0];
    for (int i = 2; i <= n; ++i) {
        size_t o = (size_t)(i - 1) * s, om = (size_t)(i - 2) * s;
        float p = 1.0f / (b[o] + a[o] * q[om]);
        q[o] = -c[o] * p;
        d[o] = (d[o] - a[o] * d[om]) * p;
    }
    for (int i = n - 1; i >= 1; --i) {
        size_t o = (size_t)(i - 1) * s, op = (size_t)i * s;
        d[o] = d[o] + q[o] * d[op];
    }
}

// Workspace: five planes of (nz) levels over (ny+1)*(nx+1) columns.
#define SMS_WS_SETUP(col)                                                   \
    size_t ncol = (size_t)(ny + 1) * (size_t)(nx + 1);                      \
    size_t s = ncol;                                                         \
    float* A = ws + (size_t)0 * nz * ncol + (col);                          \
    float* B = ws + (size_t)1 * nz * ncol + (col);                          \
    float* C = ws + (size_t)2 * nz * ncol + (col);                          \
    float* D = ws + (size_t)3 * nz * ncol + (col);                          \
    float* Q = ws + (size_t)4 * nz * ncol + (col);
#define WS(arr, k) (arr)[(size_t)((k) - 1) * s]

// vertical_diffusion_implicit, u (:7754-7864): one thread per u column.
// Periodic x: i = 0 .. nx (the duplicate east face computes the same
// words as face 0); specified/nested: ids+1 .. ide-1.
extern "C" __global__
void sms_vdiff_u(float* ru_tend, const float* u, const float* v,
                 const float* rho, const float* xkmv, const float* xkmv_meso,
                 const float* gamu, const float* rdz, const float* fnm,
                 const float* fnp, const float* dnw, const float* c1h,
                 const float* c2h, const float* mut, const float* ust,
                 float dt, float* ws,
                 int nz, int ny, int nx, int bx, int by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > nx || j >= ny) return;
    if (i < (bx ? 1 : 0) || i > (bx ? nx - 1 : nx)) return;
    if (j < (by ? 1 : 0) || j > (by ? ny - 2 : ny - 1)) return;
    const int kts = 1, ktf = nz;
    int ic = sms_wx(i, nx, bx), im = sms_wx(i - 1, nx, bx);
    SMS_WS_SETUP((size_t)j * (nx + 1) + i)

#define FNM(k) fnm[(k) - 1]
#define FNP(k) fnp[(k) - 1]
    // xkxavg_m / xkxavg_s (:7774-7795), zero at kts and ktf+1.
#define RHOAVG_U(k) (0.5f * (FNM(k) * (SMS_M(rho, k, j, ic) + SMS_M(rho, k, j, im)) \
                     + FNP(k) * (SMS_M(rho, k - 1, j, ic) + SMS_M(rho, k - 1, j, im))))
#define XKX_M(k) (((k) == kts || (k) == ktf + 1) ? 0.0f : (0.5f * RHOAVG_U(k) \
                  * (FNM(k) * (SMS_M(xkmv_meso, k, j, ic) + SMS_M(xkmv_meso, k, j, im)) \
                     + FNP(k) * (SMS_M(xkmv_meso, k - 1, j, ic) + SMS_M(xkmv_meso, k - 1, j, im)))))
#define XKX_S(k) (((k) == kts || (k) == ktf + 1) ? 0.0f : (0.5f * RHOAVG_U(k) \
                  * (FNM(k) * (SMS_M(xkmv, k, j, ic) + SMS_M(xkmv, k, j, im)) \
                     + FNP(k) * (SMS_M(xkmv, k - 1, j, ic) + SMS_M(xkmv, k - 1, j, im)))))
#define RDZ_U(k) (2.0f / (1.0f / SMS_M(rdz, k, j, ic) + 1.0f / SMS_M(rdz, k, j, im)))
#define MUAVG_U(k) (0.5f * ((c1h[(k) - 1] * SMS_P(mut, j, ic) + c2h[(k) - 1]) \
                           + (c1h[(k) - 1] * SMS_P(mut, j, im) + c2h[(k) - 1])))

    float gamuavg = 0.5f * (SMS_P(gamu, j, ic) + SMS_P(gamu, j, im));
    float vsum = SMS_V(v, kts, j, ic) + SMS_V(v, kts, j + 1, ic)
               + SMS_V(v, kts, j, im) + SMS_V(v, kts, j + 1, im);
    float V0_u = sqrtf(sms_powi2(SMS_U(u, kts, j, i))
                       + sms_powi2(vsum / 4.0f)) + SMS_EPS15;
    float ustar = 0.5f * (SMS_P(ust, j, ic) + SMS_P(ust, j, im));
    float tao_xz = ustar * ustar * (SMS_M(rho, kts, j, ic) + SMS_M(rho, kts, j, im))
                 / (2.0f * V0_u);

    int k = kts;
    float rdz_w = -SMS_G / (dnw[k - 1]) / MUAVG_U(k);
    WS(A, k) = 0.0f;
    WS(B, k) = 1.0f + rdz_w * (RDZ_U(k + 1) * XKX_S(k + 1) + tao_xz) * dt;
    WS(C, k) = -rdz_w * RDZ_U(k + 1) * XKX_S(k + 1) * dt;
    WS(D, k) = SMS_U(u, kts, j, i);
    for (k = kts + 1; k <= ktf - 1; ++k) {
        rdz_w = -SMS_G / (dnw[k - 1]) / MUAVG_U(k);
        WS(A, k) = -rdz_w * RDZ_U(k) * XKX_S(k) * dt;
        WS(B, k) = 1.0f + rdz_w * (RDZ_U(k + 1) * XKX_S(k + 1)
                                   + RDZ_U(k) * XKX_S(k)) * dt;
        WS(C, k) = -rdz_w * RDZ_U(k + 1) * XKX_S(k + 1) * dt;
        WS(D, k) = -rdz_w * (XKX_M(k + 1) - XKX_M(k)) * dt * gamuavg
                 + SMS_U(u, k, j, i);
    }
    WS(A, ktf) = 0.0f;
    WS(B, ktf) = 1.0f;
    WS(C, ktf) = 0.0f;
    WS(D, ktf) = SMS_U(u, ktf, j, i);
    sms_tridiag(ktf, A, B, C, D, Q, s);
    for (k = kts; k <= ktf; ++k) {
        SMS_U(ru_tend, k, j, i) = SMS_U(ru_tend, k, j, i)
            + MUAVG_U(k) * (WS(D, k) - SMS_U(u, k, j, i)) / dt;
    }
#undef RHOAVG_U
#undef XKX_M
#undef XKX_S
#undef RDZ_U
#undef MUAVG_U
}

// vertical_diffusion_implicit, v (:7869-7975).  Periodic y: j = 0 .. ny;
// specified/nested: jds+1 .. jde-1.
extern "C" __global__
void sms_vdiff_v(float* rv_tend, const float* u, const float* v,
                 const float* rho, const float* xkmv, const float* xkmv_meso,
                 const float* gamv, const float* rdz, const float* fnm,
                 const float* fnp, const float* dnw, const float* c1h,
                 const float* c2h, const float* mut, const float* ust,
                 float dt, float* ws,
                 int nz, int ny, int nx, int bx, int by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j > ny) return;
    if (i < (bx ? 1 : 0) || i > (bx ? nx - 2 : nx - 1)) return;
    if (j < (by ? 1 : 0) || j > (by ? ny - 1 : ny)) return;
    const int kts = 1, ktf = nz;
    int jc = by ? j : PERIODIC(j, ny), jm = by ? j - 1 : PERIODIC(j - 1, ny);
    SMS_WS_SETUP((size_t)j * (nx + 1) + i)

#define RHOAVG_V(k) (0.5f * (FNM(k) * (SMS_M(rho, k, jc, i) + SMS_M(rho, k, jm, i)) \
                     + FNP(k) * (SMS_M(rho, k - 1, jc, i) + SMS_M(rho, k - 1, jm, i))))
#define XKX_M(k) (((k) == kts || (k) == ktf + 1) ? 0.0f : (0.5f * RHOAVG_V(k) \
                  * (FNM(k) * (SMS_M(xkmv_meso, k, jc, i) + SMS_M(xkmv_meso, k, jm, i)) \
                     + FNP(k) * (SMS_M(xkmv_meso, k - 1, jc, i) + SMS_M(xkmv_meso, k - 1, jm, i)))))
#define XKX_S(k) (((k) == kts || (k) == ktf + 1) ? 0.0f : (0.5f * RHOAVG_V(k) \
                  * (FNM(k) * (SMS_M(xkmv, k, jc, i) + SMS_M(xkmv, k, jm, i)) \
                     + FNP(k) * (SMS_M(xkmv, k - 1, jc, i) + SMS_M(xkmv, k - 1, jm, i)))))
#define RDZ_V(k) (2.0f / (1.0f / SMS_M(rdz, k, jc, i) + 1.0f / SMS_M(rdz, k, jm, i)))
#define MUAVG_V(k) (0.5f * ((c1h[(k) - 1] * SMS_P(mut, jc, i) + c2h[(k) - 1]) \
                           + (c1h[(k) - 1] * SMS_P(mut, jm, i) + c2h[(k) - 1])))

    float gamvavg = 0.5f * (SMS_P(gamv, jc, i) + SMS_P(gamv, jm, i));
    // u faces i and i+1 on rows j and j-1 (u is (nz, ny, nx+1)).
    float usum = SMS_U(u, kts, jc, i) + SMS_U(u, kts, jm, i)
               + SMS_U(u, kts, jc, i + 1) + SMS_U(u, kts, jm, i + 1);
    float V0_v = sqrtf(sms_powi2(SMS_V(v, kts, j, i))
                       + sms_powi2(usum / 4.0f)) + SMS_EPS15;
    float ustar = 0.5f * (SMS_P(ust, jc, i) + SMS_P(ust, jm, i));
    float tao_yz = ustar * ustar * (SMS_M(rho, kts, jc, i) + SMS_M(rho, kts, jm, i))
                 / (2.0f * V0_v);

    int k = kts;
    float rdz_w = -SMS_G / (dnw[k - 1]) / MUAVG_V(k);
    WS(A, k) = 0.0f;
    WS(B, k) = 1.0f + rdz_w * (RDZ_V(k + 1) * XKX_S(k + 1) + tao_yz) * dt;
    WS(C, k) = -rdz_w * RDZ_V(k + 1) * XKX_S(k + 1) * dt;
    WS(D, k) = SMS_V(v, kts, j, i);
    for (k = kts + 1; k <= ktf - 1; ++k) {
        rdz_w = -SMS_G / (dnw[k - 1]) / MUAVG_V(k);
        WS(A, k) = -rdz_w * RDZ_V(k) * XKX_S(k) * dt;
        WS(B, k) = 1.0f + rdz_w * (RDZ_V(k + 1) * XKX_S(k + 1)
                                   + RDZ_V(k) * XKX_S(k)) * dt;
        WS(C, k) = -rdz_w * RDZ_V(k + 1) * XKX_S(k + 1) * dt;
        WS(D, k) = -rdz_w * (XKX_M(k + 1) - XKX_M(k)) * dt * gamvavg
                 + SMS_V(v, k, j, i);
    }
    WS(A, ktf) = 0.0f;
    WS(B, ktf) = 1.0f;
    WS(C, ktf) = 0.0f;
    WS(D, ktf) = SMS_V(v, ktf, j, i);
    sms_tridiag(ktf, A, B, C, D, Q, s);
    for (k = kts; k <= ktf; ++k) {
        SMS_V(rv_tend, k, j, i) = SMS_V(rv_tend, k, j, i)
            + MUAVG_V(k) * (WS(D, k) - SMS_V(v, k, j, i)) / dt;
    }
#undef RHOAVG_V
#undef XKX_M
#undef XKX_S
#undef RDZ_V
#undef MUAVG_V
}

// vertical_diffusion_implicit, w (:7980-8023): w levels kts+1..ktf with
// xkmh (WRF's choice) times rho as the mixing coefficient.
extern "C" __global__
void sms_vdiff_w(float* rw_tend, const float* w, const float* rho,
                 const float* xkmh, const float* rdzw, const float* dn,
                 const float* c1f, const float* c2f, const float* mut,
                 float dt, float* ws,
                 int nz, int ny, int nx, int bx, int by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny || !sms_mass_in(i, j, nx, ny, bx, by)) return;
    const int kts = 1, ktf = nz;
    SMS_WS_SETUP((size_t)j * (nx + 1) + i)
    float mu = SMS_P(mut, j, i);
#define XW(k) (SMS_M(xkmh, k, j, i) * SMS_M(rho, k, j, i))
    for (int k = kts + 1; k <= ktf; ++k) {
        float rdz_z = -SMS_G / dn[k - 1] / (c1f[k - 1] * mu + c2f[k - 1]);
        WS(A, k - 1) = -rdz_z * SMS_M(rdzw, k - 1, j, i) * XW(k - 1) * dt;
        WS(B, k - 1) = 1.0f + rdz_z * (SMS_M(rdzw, k, j, i) * XW(k)
                                       + SMS_M(rdzw, k - 1, j, i) * XW(k - 1)) * dt;
        WS(C, k - 1) = -rdz_z * SMS_M(rdzw, k, j, i) * XW(k) * dt;
        WS(D, k - 1) = SMS_W(w, k, j, i);
    }
#undef XW
    sms_tridiag(ktf - 1, A, B, C, D, Q, s);
    for (int k = kts + 1; k <= ktf; ++k) {
        SMS_W(rw_tend, k, j, i) = SMS_W(rw_tend, k, j, i)
            + (c1f[k - 1] * mu + c2f[k - 1]) * (WS(D, k - 1) - SMS_W(w, k, j, i)) / dt;
    }
}

// vertical_diffusion_implicit, theta (:8028-8146).  ``var`` is WRF's
// var_mix (t_2 - t_base, or t_2 with mix_full_fields), the nonlocal flux
// enters the right-hand side and the surface heat flux the first level.
// isfflx 0 and 2 re-write HFX from tke_heat_flux (the same word
// nonlocal_flux wrote).
extern "C" __global__
void sms_vdiff_th(float* rt_tend, const float* var, const float* th,
                  const float* rho, const float* xkhv, const float* nlflux,
                  const float* rdz, const float* fnm, const float* fnp,
                  const float* dnw, const float* c1h, const float* c2h,
                  const float* mut, float* hfx, const float* qfx,
                  const float* qv, int has_qv, int use_theta_m,
                  int isfflx, float tke_heat_flux, float dt, float* ws,
                  int nz, int ny, int nx, int bx, int by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny || !sms_mass_in(i, j, nx, ny, bx, by)) return;
    const int kts = 1, ktf = nz;
    SMS_WS_SETUP((size_t)j * (nx + 1) + i)
    float mu = SMS_P(mut, j, i);
    float qv1 = has_qv ? SMS_M(qv, kts, j, i) : 0.0f;
#define RHOW(k) (FNM(k) * SMS_M(rho, k, j, i) + FNP(k) * SMS_M(rho, k - 1, j, i))
#define XKX(k) (((k) == kts || (k) == ktf + 1) ? 0.0f \
                : ((FNM(k) * SMS_M(xkhv, k, j, i) + FNP(k) * SMS_M(xkhv, k - 1, j, i)) * RHOW(k)))
#define NLR(k) (SMS_W(nlflux, k, j, i) * RHOW(k))
    if (isfflx == 0 || isfflx == 2) {
        float cpm = SMS_CP * (1.0f + 0.8f * qv1);
        SMS_P(hfx, j, i) = tke_heat_flux * cpm * SMS_M(rho, 1, j, i);
    }
    float cpm = SMS_CP * (1.0f + 0.8f * qv1);
    int k = kts;
    float rdz_w = -SMS_G / dnw[k - 1] / (c1h[k - 1] * mu + c2h[k - 1]);
    WS(A, k) = 0.0f;
    WS(B, k) = 1.0f + rdz_w * (SMS_M(rdz, k + 1, j, i) * XKX(k + 1)) * dt;
    WS(C, k) = -rdz_w * SMS_M(rdz, k + 1, j, i) * XKX(k + 1) * dt;
    float qfac;
    if (use_theta_m == 0) {
        WS(D, k) = SMS_M(var, kts, j, i)
                 - dt * rdz_w * NLR(k + 1)
                 + dt * rdz_w * SMS_P(hfx, j, i) / cpm;
    } else {
        qfac = 1.0f + SMS_RVOVRD * qv1;
        WS(D, k) = SMS_M(var, kts, j, i)
                 - dt * rdz_w * NLR(k + 1) * qfac
                 + dt * rdz_w * (qfac * SMS_P(hfx, j, i) / cpm
                                 + 1.61f * SMS_M(th, kts, j, i) * SMS_P(qfx, j, i));
    }
    if (use_theta_m == 1) {
        qfac = 1.0f + SMS_RVOVRD * qv1;
    } else {
        qfac = 1.0f;
    }
    for (k = kts + 1; k <= ktf - 1; ++k) {
        rdz_w = -SMS_G / dnw[k - 1] / (c1h[k - 1] * mu + c2h[k - 1]);
        WS(A, k) = -rdz_w * SMS_M(rdz, k, j, i) * XKX(k) * dt;
        WS(B, k) = 1.0f + rdz_w * (SMS_M(rdz, k + 1, j, i) * XKX(k + 1)
                                   + SMS_M(rdz, k, j, i) * XKX(k)) * dt;
        WS(C, k) = -rdz_w * SMS_M(rdz, k + 1, j, i) * XKX(k + 1) * dt;
        WS(D, k) = -rdz_w * (NLR(k + 1) - NLR(k)) * qfac * dt
                 + SMS_M(var, k, j, i);
    }
    WS(A, ktf) = 0.0f;
    WS(B, ktf) = 1.0f;
    WS(C, ktf) = 0.0f;
    WS(D, ktf) = SMS_M(var, ktf, j, i);
    sms_tridiag(ktf, A, B, C, D, Q, s);
    for (k = kts; k <= ktf; ++k) {
        SMS_M(rt_tend, k, j, i) = SMS_M(rt_tend, k, j, i)
            + (c1h[k - 1] * mu + c2h[k - 1]) * (WS(D, k) - SMS_M(var, k, j, i)) / dt;
    }
#undef XKX
#undef NLR
}

// vertical_diffusion_implicit, TKE (:8151-8205): diffusion with 2*xkmv
// (doubled again in the coefficients, as written) and the implicit
// dissipation beta = 1.5*sqrt(e)/L_DISS with its explicit half on the
// right-hand side; the top level is held at zero.
extern "C" __global__
void sms_vdiff_tke(float* tke_tend, const float* tke, const float* rho,
                   const float* xkmv, const float* l_diss, const float* rdz,
                   const float* fnm, const float* fnp, const float* dnw,
                   const float* c1h, const float* c2h, const float* mut,
                   float dt, float* ws,
                   int nz, int ny, int nx, int bx, int by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny || !sms_mass_in(i, j, nx, ny, bx, by)) return;
    const int kts = 1, ktf = nz;
    SMS_WS_SETUP((size_t)j * (nx + 1) + i)
    float mu = SMS_P(mut, j, i);
#define XKX(k) (((k) == kts || (k) == ktf + 1) ? 0.0f \
                : ((FNM(k) * 2.0f * SMS_M(xkmv, k, j, i) + FNP(k) * 2.0f * SMS_M(xkmv, k - 1, j, i)) \
                   * (FNM(k) * SMS_M(rho, k, j, i) + FNP(k) * SMS_M(rho, k - 1, j, i))))
    for (int k = kts; k <= ktf - 1; ++k) {
        float rdz_w = -SMS_G / dnw[k - 1] / (c1h[k - 1] * mu + c2h[k - 1]);
        float ld = SMS_M(l_diss, k, j, i);
        float tk = SMS_M(tke, k, j, i);
        float beta;
        if (ld != 0.0f) {
            beta = 1.5f * sqrtf(tk) / ld;
        } else {
            beta = 0.0f;
        }
        WS(A, k) = -2.0f * XKX(k) * dt * rdz_w * SMS_M(rdz, k, j, i);
        WS(B, k) = 1.0f + 2.0f * dt * rdz_w * (SMS_M(rdz, k, j, i) * XKX(k)
                                              + SMS_M(rdz, k + 1, j, i) * XKX(k + 1))
                 + dt * beta;
        WS(C, k) = -2.0f * XKX(k + 1) * dt * SMS_M(rdz, k + 1, j, i) * rdz_w;
        float d = tk;
        if (ld != 0.0f) {
            d = d + 0.5f * dt * gfk_pow(tk, 1.5f) / ld;
        }
        WS(D, k) = d;
    }
#undef XKX
    WS(A, ktf) = 0.0f;
    WS(B, ktf) = 1.0f;
    WS(C, ktf) = 0.0f;
    WS(D, ktf) = 0.0f;
    sms_tridiag(ktf, A, B, C, D, Q, s);
    for (int k = kts; k <= ktf; ++k) {
        SMS_M(tke_tend, k, j, i) = SMS_M(tke_tend, k, j, i)
            + (c1h[k - 1] * mu + c2h[k - 1]) * (WS(D, k) - SMS_M(tke, k, j, i)) / dt;
    }
}

// vertical_diffusion_implicit, one scalar row (:8210-8475): the moist
// loop (with the surface moisture flux QFX on the water-vapour row), the
// scalar, tracer and chem loops.  ``base`` (nullable, has_base) is WRF's
// qv_base subtracted from water vapour unless mix_full_fields.
extern "C" __global__
void sms_vdiff_scalar(float* tend, const float* field, const float* base,
                      int has_base, const float* rho, const float* xkhv,
                      const float* rdz, const float* fnm, const float* fnp,
                      const float* dnw, const float* c1h, const float* c2h,
                      const float* mut, const float* sfc_flux, int has_flux,
                      float dt, float* ws,
                      int nz, int ny, int nx, int bx, int by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny || !sms_mass_in(i, j, nx, ny, bx, by)) return;
    const int kts = 1, ktf = nz;
    SMS_WS_SETUP((size_t)j * (nx + 1) + i)
    float mu = SMS_P(mut, j, i);
#define VAR(k) (has_base ? (SMS_M(field, k, j, i) - base[(k) - 1]) : SMS_M(field, k, j, i))
#define XKX(k) (((k) == kts || (k) == ktf + 1) ? 0.0f \
                : ((FNM(k) * SMS_M(xkhv, k, j, i) + FNP(k) * SMS_M(xkhv, k - 1, j, i)) \
                   * (FNM(k) * SMS_M(rho, k, j, i) + FNP(k) * SMS_M(rho, k - 1, j, i))))
    int k = kts;
    float rdz_w = -SMS_G / dnw[k - 1] / (c1h[k - 1] * mu + c2h[k - 1]);
    WS(A, k) = 0.0f;
    WS(B, k) = 1.0f + rdz_w * (SMS_M(rdz, k + 1, j, i) * XKX(k + 1)) * dt;
    WS(C, k) = -rdz_w * SMS_M(rdz, k + 1, j, i) * XKX(k + 1) * dt;
    if (has_flux) {
        WS(D, k) = VAR(k) + dt * rdz_w * SMS_P(sfc_flux, j, i);
    } else {
        WS(D, k) = VAR(k);
    }
    for (k = kts + 1; k <= ktf - 1; ++k) {
        rdz_w = -SMS_G / dnw[k - 1] / (c1h[k - 1] * mu + c2h[k - 1]);
        WS(A, k) = -rdz_w * SMS_M(rdz, k, j, i) * XKX(k) * dt;
        WS(B, k) = 1.0f + rdz_w * (SMS_M(rdz, k + 1, j, i) * XKX(k + 1)
                                   + SMS_M(rdz, k, j, i) * XKX(k)) * dt;
        WS(C, k) = -rdz_w * SMS_M(rdz, k + 1, j, i) * XKX(k + 1) * dt;
        WS(D, k) = VAR(k);
    }
    WS(A, ktf) = 0.0f;
    WS(B, ktf) = 1.0f;
    WS(C, ktf) = 0.0f;
    WS(D, ktf) = VAR(ktf);
    sms_tridiag(ktf, A, B, C, D, Q, s);
    for (k = kts; k <= ktf; ++k) {
        SMS_M(tend, k, j, i) = SMS_M(tend, k, j, i)
            + (c1h[k - 1] * mu + c2h[k - 1]) * (WS(D, k) - VAR(k)) / dt;
    }
#undef VAR
#undef XKX
}
#undef FNM
#undef FNP

// ---------------------------------------------------------------------------
// The WRF-native operands the routines above read, from woof's state, as
// phy_prep (module_big_step_utilities_em.F:4730-4973) and
// compute_diff_metrics (module_diffusion_em.F:6882-7130) form them.
//
// sms_prep_mass: TH (th_phy = t_2 + t0), T2 (t_2, WRF's theta - 300), RHO
// (1./alt*(1.+qv)), U_PHY/V_PHY (the mass-centred winds), RDZW and RDZ.
// woof stores theta as thp about thb; t_2 is thp itself where thb is
// 300 K (the strict and wrfinput carriers) and (thb - 300) + thp
// elsewhere.
extern "C" __global__
void sms_prep_mass(const float* u, const float* v, const float* php,
                   const float* phb, int phb3d, const float* thp,
                   const float* thb, int thb3d, const float* alt,
                   const float* qv, int has_qv,
                   float* th, float* t2, float* rho, float* u_phy,
                   float* v_phy, float* rdzw, float* rdz,
                   int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int kc = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || kc >= nz) return;
    int k = kc + 1;
    size_t h = ((size_t)kc * ny + j) * nx + i;
    float base = thb3d ? thb[h] : thb[kc];
    float tb = base - 300.0f;
    float tt = (tb == 0.0f) ? thp[h] : tb + thp[h];
    t2[h] = tt;
    th[h] = tt + 300.0f;
    float q = has_qv ? qv[h] : 0.0f;
    rho[h] = 1.0f / alt[h] * (1.0f + q);
    u_phy[h] = 0.5f * (SMS_U(u, k, j, i) + SMS_U(u, k, j, i + 1));
    v_phy[h] = 0.5f * (SMS_V(v, k, j, i) + SMS_V(v, k, j + 1, i));
    // compute_diff_metrics adds ph + phb in that order.
#define ZW(kk) ((php[((size_t)((kk) - 1) * ny + j) * nx + i] \
                 + (phb3d ? phb[((size_t)((kk) - 1) * ny + j) * nx + i] : phb[(kk) - 1])) / SMS_G)
    rdzw[h] = 1.0f / (ZW(k + 1) - ZW(k));
    if (k == 1) {
        rdz[h] = 2.0f / (ZW(2) - ZW(1));
    } else {
        rdz[h] = 2.0f / (ZW(k + 1) - ZW(k - 1));
    }
#undef ZW
}

// sms_prep_slope: ZX on (nz+1, ny, nx+1) u faces and ZY on (nz+1, ny+1, nx)
// v faces, compute_diff_metrics' two-pass rdx*(phb - phb)/g +
// rdx*(ph - ph)/g, with the physical-boundary words that routine and
// set_physical_bc3d('e'/'f') leave (module_first_rk_step_part2.F:478-489):
//   specified/nested: the west/south outer face is compute_diff_metrics'
//     zero; the east/north outer face is set_physical_bc3d's open copy of
//     the last interior face, at every level (module_bc.F:893-908);
//   periodic: the outer faces are the periodic branch's words at levels
//     1..ktf; at the top level kde the face ids is never written (the main
//     loop starts at ids+1, the periodic branch stops at ktf), and
//     set_physical_bc3d copies that zero to ide.
extern "C" __global__
void sms_prep_slope(const float* php, const float* phb, int phb3d,
                    float rdx, float rdy, float* zx, float* zy,
                    int nz, int ny, int nx, int bx, int by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int kc = blockIdx.z * blockDim.z + threadIdx.z;
    if (kc > nz) return;
    int k = kc + 1;
#define PH(jj, ii) php[((size_t)kc * ny + (jj)) * nx + (ii)]
#define PB(jj, ii) (phb3d ? phb[((size_t)kc * ny + (jj)) * nx + (ii)] : phb[kc])
    if (i <= nx && j < ny) {
        float val;
        int ii = (bx && i == nx) ? nx - 1 : i;
        if ((bx && i == 0) || (!bx && (i == 0 || i == nx) && k == nz + 1)) {
            val = 0.0f;
        } else {
            int ic = PERIODIC(ii, nx), im = PERIODIC(ii - 1, nx);
            val = rdx * (PB(j, ic) - PB(j, im)) / SMS_G;
            val = val + rdx * (PH(j, ic) - PH(j, im)) / SMS_G;
        }
        zx[((size_t)kc * ny + j) * (nx + 1) + i] = val;
    }
    if (i < nx && j <= ny) {
        float val;
        int jj = (by && j == ny) ? ny - 1 : j;
        if ((by && j == 0) || (!by && (j == 0 || j == ny) && k == nz + 1)) {
            val = 0.0f;
        } else {
            int jc = PERIODIC(jj, ny), jm = PERIODIC(jj - 1, ny);
            val = rdy * (PB(jc, i) - PB(jm, i)) / SMS_G;
            val = val + rdy * (PH(jc, i) - PH(jm, i)) / SMS_G;
        }
        zy[((size_t)kc * (ny + 1) + j) * nx + i] = val;
    }
#undef PH
#undef PB
}

// sms_prep_p8w: phy_prep's surface and top extrapolations of p8w/t8w
// (:4895-4921), the only levels km_opt = 5 reads (thetasfc, thetatop).
// p is woof's full pressure (WRF p_phy = p + pb); t_phy = th*pi.
extern "C" __global__
void sms_prep_p8w(const float* p, const float* th, const float* php,
                  const float* phb, int phb3d,
                  float* p8w1, float* t8w1, float* p8wt, float* t8wt,
                  int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
#define ZAW(kk) (((phb3d ? phb[((size_t)((kk) - 1) * ny + j) * nx + i] : phb[(kk) - 1]) \
                 + php[((size_t)((kk) - 1) * ny + j) * nx + i]) / SMS_G)
#define PP(kk) SMS_M(p, kk, j, i)
#define TPHY(kk) (SMS_M(th, kk, j, i) * gfk_pow(PP(kk) / SMS_P1000, SMS_RCP))
    const int kde = nz + 1, k_end = nz;
    float z0 = ZAW(1);
    float z1 = 0.5f * (ZAW(1) + ZAW(2));
    float z2 = 0.5f * (ZAW(2) + ZAW(3));
    float w1 = (z0 - z2) / (z1 - z2);
    float w2 = 1.0f - w1;
    SMS_P(p8w1, j, i) = w1 * PP(1) + w2 * PP(2);
    SMS_P(t8w1, j, i) = w1 * TPHY(1) + w2 * TPHY(2);
    z0 = ZAW(kde);
    z1 = 0.5f * (ZAW(k_end) + ZAW(k_end + 1));
    z2 = 0.5f * (ZAW(k_end - 1) + ZAW(k_end));
    w1 = (z0 - z2) / (z1 - z2);
    w2 = 1.0f - w1;
    SMS_P(p8wt, j, i) = gfk_exp(w1 * gfk_log(PP(kde - 1)) + w2 * gfk_log(PP(kde - 2)));
    SMS_P(t8wt, j, i) = w1 * TPHY(kde - 1) + w2 * TPHY(kde - 2);
#undef ZAW
#undef PP
#undef TPHY
}
