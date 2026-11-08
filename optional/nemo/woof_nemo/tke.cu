// WOOF-OPTIN: nemo
// Derived from NEMO 5.0.2 (CeCILL v2.0); see optional/nemo/LICENSE and
// NOTICE.md.  Not part of the default build or of any wheel.
//
// NEMO's TKE vertical mixing: src/OCE/ZDF/zdfsh2.F90 (zdf_sh2, no Stokes
// shear) and src/OCE/ZDF/zdftke.F90 (tke_tke, tke_avn), for the
// configurations zdf_tke_port() in tke.py accepts: no sea-ice model
// (nn_eice = 0), no ice-shelf cavities, bottom friction on, Langmuir cells
// from the surface stress, nn_etau 0 or 1, nn_htau 0 or 1, nn_mxl 0-3,
// nn_pdl 0 or 1, ln_mxl0 either way.  NEMO computes in double; every
// expression is NEMO's in its order, with no contraction; EXP is glibc's
// binary64 exp, SIN WOOF's correctly rounded binary64 sin.  Fields are
// (jpk, jpj, jpi) C order with NEMO's one-cell halo; one thread per
// interior column (Nis0..Nie0, Njs0..Nje0, 1-based).

struct TkeConf {
    double rn_ediff, rn_ediss, rn_ebb, rn_emin, rn_emin0, rn_bshear, rn_mxl0, rn_lc, rn_efr;
    double rn_Dt, rho0, grav, vkarmn, rpi, rsmall, ri_cri, rmxl_min;
    int nn_mxl, nn_pdl, ln_mxl0, ln_lc, nn_etau;
};

#define I3(i, j, k) ((size_t)((k) - 1) * n2 + (size_t)((j) - 1) * nx + ((i) - 1))
#define I2(i, j) ((size_t)((j) - 1) * nx + ((i) - 1))
#define MAXD(a, b) ((a) >= (b) ? (a) : (b))
#define MIND(a, b) ((a) <= (b) ? (a) : (b))

// zdf_sh2 (:31-93): the shear production at w-points.
__device__ static double tke_sh2u(const double* avm, const double* uu1, const double* uu2,
                                  const double* e3uw, const double* wumask,
                                  int i, int j, int k, int nx, size_t n2)
{
    return ( avm[I3(i + 1, j, k)] + avm[I3(i, j, k)] )
         * (   uu2[I3(i, j, k - 1)] -   uu2[I3(i, j, k)] )
         * (   uu1[I3(i, j, k - 1)] -   uu1[I3(i, j, k)] )
         / ( e3uw[I3(i, j, k)] * e3uw[I3(i, j, k)] )
         * wumask[I3(i, j, k)];
}
__device__ static double tke_sh2v(const double* avm, const double* vv1, const double* vv2,
                                  const double* e3vw, const double* wvmask,
                                  int i, int j, int k, int nx, size_t n2)
{
    return ( avm[I3(i, j + 1, k)] + avm[I3(i, j, k)] )
         * (   vv2[I3(i, j, k - 1)] -   vv2[I3(i, j, k)] )
         * (   vv1[I3(i, j, k - 1)] -   vv1[I3(i, j, k)] )
         / ( e3vw[I3(i, j, k)] * e3vw[I3(i, j, k)] )
         * wvmask[I3(i, j, k)];
}

extern "C" __global__
void nemo_zdf_sh2(const double* __restrict__ avm, const double* __restrict__ uu1,
                  const double* __restrict__ uu2, const double* __restrict__ vv1,
                  const double* __restrict__ vv2, const double* __restrict__ e3uw,
                  const double* __restrict__ e3vw, const double* __restrict__ wumask,
                  const double* __restrict__ wvmask, const double* __restrict__ umask,
                  const double* __restrict__ vmask, double* __restrict__ sh2,
                  int nx, int ny, int jpk)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int ni = nx - 2, nj = ny - 2;
    if (t >= ni * nj) return;
    int i = t % ni + 2, j = t / ni + 2;
    size_t n2 = (size_t)nx * ny;
    for (int k = 2; k <= jpk - 1; ++k) {
        double zu0 = tke_sh2u(avm, uu1, uu2, e3uw, wumask, i - 1, j, k, nx, n2);
        double zu1 = tke_sh2u(avm, uu1, uu2, e3uw, wumask, i, j, k, nx, n2);
        double zv0 = tke_sh2v(avm, vv1, vv2, e3vw, wvmask, i, j - 1, k, nx, n2);
        double zv1 = tke_sh2v(avm, vv1, vv2, e3vw, wvmask, i, j, k, nx, n2);
        sh2[I3(i, j, k)] = 0.25 * (   ( zu0 + zu1 ) * ( 2. - umask[I3(i - 1, j, k)] * umask[I3(i, j, k)] )
                                    + ( zv0 + zv1 ) * ( 2. - vmask[I3(i, j - 1, k)] * vmask[I3(i, j, k)] )   );
    }
    sh2[I3(i, j, 1)] = 0.0;
    sh2[I3(i, j, jpk)] = 0.0;
}

// zdf_tke_init's htau (:862-869).
extern "C" __global__
void nemo_tke_htau(const double* __restrict__ gphit, double* __restrict__ htau,
                   double rpi, int nn_htau, int n2)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n2) return;
    if (nn_htau == 0) htau[c] = 10.0;
    else htau[c] = MAXD(  0.5, MIND( 30.0, 45.0* fabs( wcr_sin( rpi/180.0 * gphit[c] ) ) )   );
}

#define TKE_KMAX 128

extern "C" __global__
void nemo_zdf_tke(const double* __restrict__ cfd, int nn_mxl, int nn_pdl, int ln_mxl0,
                  int ln_lc, int nn_etau, const double* __restrict__ sh2, double* __restrict__ avm,
                  double* __restrict__ avt, double* __restrict__ en, double* __restrict__ dissl,
                  const double* __restrict__ rn2, const double* __restrict__ rn2b,
                  const double* __restrict__ taum, const double* __restrict__ rCdU_bot,
                  const double* __restrict__ uu1, const double* __restrict__ vv1,
                  const double* __restrict__ umask, const double* __restrict__ vmask,
                  const double* __restrict__ tmask, const double* __restrict__ wmask,
                  const double* __restrict__ ssmask, const int* __restrict__ mbkt,
                  const double* __restrict__ e3t, const double* __restrict__ e3w,
                  const double* __restrict__ gdepw, const double* __restrict__ htau,
                  const double* __restrict__ avmb, const double* __restrict__ avtb,
                  const double* __restrict__ avtb_2d, int nx, int ny, int jpk)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int ni = nx - 2, nj = ny - 2;
    if (t >= ni * nj) return;
    int i = t % ni + 2, j = t / ni + 2;
    size_t n2 = (size_t)nx * ny;
    int jpkm1 = jpk - 1;
    TkeConf cf;   // tke.py's TKE_CONST order
    cf.rn_ediff = cfd[0]; cf.rn_ediss = cfd[1]; cf.rn_ebb = cfd[2]; cf.rn_emin = cfd[3];
    cf.rn_emin0 = cfd[4]; cf.rn_bshear = cfd[5]; cf.rn_mxl0 = cfd[6]; cf.rn_lc = cfd[7];
    cf.rn_efr = cfd[8]; cf.rn_Dt = cfd[9]; cf.rho0 = cfd[10]; cf.grav = cfd[11];
    cf.vkarmn = cfd[12]; cf.rpi = cfd[13]; cf.rsmall = cfd[14]; cf.ri_cri = cfd[15];
    cf.rmxl_min = cfd[16];
    cf.nn_mxl = nn_mxl; cf.nn_pdl = nn_pdl; cf.ln_mxl0 = ln_mxl0; cf.ln_lc = ln_lc;
    cf.nn_etau = nn_etau;
    if (jpk >= TKE_KMAX) return;
    double zdiag[TKE_KMAX], zd_up[TKE_KMAX], zd_lw[TKE_KMAX], zpelc[TKE_KMAX], pdlr[TKE_KMAX];
    // ---- tke_tke (:193-522) ----
    double zrhoa = (double)1.22f, zcdrag = (double)1.5e-3f;   // REAL(wp) set from REAL literals
    double zbbrau = cf.rn_ebb / cf.rho0;
    double zfact1 = -.5 * cf.rn_Dt;
    double zfact2 = 1.5 * cf.rn_Dt * cf.rn_ediss;
    double zfact3 = 0.5 * cf.rn_ediss;
    double zcsd = (double)(0.5f * 0.016f * 0.016f) / ( zrhoa * zcdrag );
    double zice_fra = 0.0;                                    // nn_eice = 0
    en[I3(i, j, 1)] = MAXD( cf.rn_emin0, zbbrau * taum[I2(i, j)] );
    zdiag[1] = 1.0/en[I3(i, j, 1)];
    zd_lw[1] = 1.0;
    zd_up[1] = 0.0;
    {   // bottom friction (ln_drg_OFF = F)
        int kb = mbkt[I2(i, j)];
        double zmsku = ( 2. - umask[I3(i - 1, j, kb)] * umask[I3(i, j, kb)] );
        double zmskv = ( 2. - vmask[I3(i, j - 1, kb)] * vmask[I3(i, j, kb)] );
        double a = zmsku*( uu1[I3(i, j, kb)] + uu1[I3(i - 1, j, kb)] );
        double b = zmskv*( vv1[I3(i, j, kb)] + vv1[I3(i, j - 1, kb)] );
        double zebot = - 0.001875 * rCdU_bot[I2(i, j)] * sqrt( a * a + b * b );
        en[I3(i, j, kb + 1)] = MAXD( zebot, cf.rn_emin ) * ssmask[I2(i, j)];
    }
    if (cf.ln_lc) {
        double zWlc2 = zcsd * taum[I2(i, j)];
        zpelc[1] = MAXD( rn2b[I3(i, j, 1)], 0.0 ) * gdepw[I3(i, j, 1)] * e3w[I3(i, j, 1)];
        for (int k = 2; k <= jpk; ++k)
            zpelc[k] = zpelc[k - 1] + MAXD( rn2b[I3(i, j, k)], 0.0 ) * gdepw[I3(i, j, k)] * e3w[I3(i, j, k)];
        int imlc = mbkt[I2(i, j)] + 1;
        for (int k = jpkm1; k >= 2; --k)
            if (zpelc[k] > zWlc2) imlc = k;
        double zhlc = gdepw[I3(i, j, imlc)];
        double zus = sqrt( 2. * zWlc2 );
        double zus3 = MAXD( 0.0, 1.0 - zice_fra ) * zus * zus * zus * tmask[I3(i, j, 1)];
        for (int k = 2; k <= jpkm1; ++k) {
            if (zus3 != 0.0) {
                if ( gdepw[I3(i, j, k)] - zhlc < 0 && wmask[I3(i, j, k)] != 0. ) {
                    double zwlc = cf.rn_lc * wcr_sin( cf.rpi * gdepw[I3(i, j, k)] / zhlc );
                    en[I3(i, j, k)] = en[I3(i, j, k)] + cf.rn_Dt * zus3 * ( zwlc * zwlc * zwlc ) / zhlc;
                }
            }
        }
    }
    if (cf.nn_pdl == 1) {
        for (int k = 2; k <= jpkm1; ++k) {
            double zri;
            if (rn2b[I3(i, j, k)] <= 0.0) {
                zri = 0.0;
            } else {
                double zdiv = sh2[I3(i, j, k)] + cf.rn_bshear;
                if (zdiv == 0.0) zri = rn2b[I3(i, j, k)] * avm[I3(i, j, k)] / cf.rn_bshear;
                else zri = rn2b[I3(i, j, k)] * avm[I3(i, j, k)] / zdiv;
            }
            pdlr[k] = MAXD(  0.1,  cf.ri_cri / MAXD( cf.ri_cri , zri )  );
        }
    }
    for (int k = 2; k <= jpkm1; ++k) {
        double zcof = zfact1 * tmask[I3(i, j, k)];
        double zzd_up = zcof * MAXD(  avm[I3(i, j, k + 1)] + avm[I3(i, j, k)] , 2.e-5  )
                      /    (  e3t[I3(i, j, k)] * e3w[I3(i, j, k)]  );
        double zzd_lw = zcof * MAXD(  avm[I3(i, j, k)] + avm[I3(i, j, k - 1)] , 2.e-5  )
                      /    (  e3t[I3(i, j, k - 1)] * e3w[I3(i, j, k)]  );
        zd_up[k] = zzd_up;
        zd_lw[k] = zzd_lw;
        zdiag[k] = 1.0 - zzd_lw - zzd_up + zfact2 * dissl[I3(i, j, k)] * wmask[I3(i, j, k)];
        en[I3(i, j, k)] = en[I3(i, j, k)] + cf.rn_Dt * (  sh2[I3(i, j, k)]
                                                     - avt[I3(i, j, k)] * rn2[I3(i, j, k)]
                                                     + zfact3 * dissl[I3(i, j, k)] * en[I3(i, j, k)]
                                                    ) * wmask[I3(i, j, k)];
    }
    for (int k = 2; k <= jpkm1; ++k)
        zdiag[k] = zdiag[k] - zd_lw[k] * zd_up[k - 1] / zdiag[k - 1];
    for (int k = 2; k <= jpkm1; ++k)
        zd_lw[k] = en[I3(i, j, k)] - zd_lw[k] / zdiag[k - 1] * zd_lw[k - 1];
    en[I3(i, j, jpkm1)] = zd_lw[jpkm1] / zdiag[jpkm1];
    for (int k = jpk - 2; k >= 2; --k)
        en[I3(i, j, k)] = ( zd_lw[k] - zd_up[k] * en[I3(i, j, k + 1)] ) / zdiag[k];
    for (int k = 2; k <= jpkm1; ++k)
        en[I3(i, j, k)] = MAXD( en[I3(i, j, k)], cf.rn_emin ) * wmask[I3(i, j, k)];
    if (cf.nn_etau == 1) {
        for (int k = 2; k <= jpkm1; ++k)
            en[I3(i, j, k)] = en[I3(i, j, k)] + cf.rn_efr * en[I3(i, j, 1)] * glibc_exp( -gdepw[I3(i, j, k)] / htau[I2(i, j)] )
                * MAXD( 0.0, 1.0 - zice_fra ) * wmask[I3(i, j, k)] * tmask[I3(i, j, 1)];
    }
    // ---- tke_avn (:525-735) ----
    double zmxlm[TKE_KMAX], zmxld[TKE_KMAX];
    for (int k = 1; k <= jpk; ++k) { zmxlm[k] = cf.rmxl_min; zmxld[k] = cf.rmxl_min; }
    if (cf.ln_mxl0) {
        double zraug = cf.vkarmn * 2.e5 / ( cf.rho0 * cf.grav );
        zmxlm[1] = zraug * taum[I2(i, j)] * tmask[I3(i, j, 1)];
        zmxlm[1] = MAXD( cf.rn_mxl0, zmxlm[1] );
    } else {
        zmxlm[1] = cf.rn_mxl0;
    }
    for (int k = 2; k <= jpkm1; ++k) {
        double zrn2 = MAXD( rn2[I3(i, j, k)], cf.rsmall );
        zmxlm[k] = MAXD(  cf.rmxl_min,  sqrt( 2.0 * en[I3(i, j, k)] / zrn2 )  );
    }
    zmxld[1] = zmxlm[1];
    if (cf.nn_mxl == 0) {
        int kb = mbkt[I2(i, j)];
        for (int k = 2; k <= jpkm1; ++k) {
            double a = gdepw[I3(i, j, k)] - gdepw[I3(i, j, 1)];      // mikt = 1
            double b = gdepw[I3(i, j, kb + 1)] - gdepw[I3(i, j, k)];
            double zemxl = MIND( MIND( a, zmxlm[k] ), b );
            double w = wmask[I3(i, j, k)];
            double tail = MIND( zmxlm[k] , e3w[I3(i, j, k)] ) * (1 - w);
            zmxlm[k] = zemxl * w + tail;
            zmxld[k] = zemxl * w + MIND( zmxlm[k] , e3w[I3(i, j, k)] ) * (1 - w);
        }
    } else if (cf.nn_mxl == 1) {
        for (int k = 2; k <= jpkm1; ++k) {
            double zemxl = MIND( e3w[I3(i, j, k)], zmxlm[k] );
            zmxlm[k] = zemxl;
            zmxld[k] = zemxl;
        }
    } else if (cf.nn_mxl == 2) {
        for (int k = 2; k <= jpkm1; ++k)
            zmxlm[k] = MIND( zmxlm[k - 1] + e3t[I3(i, j, k - 1)], zmxlm[k] );
        for (int k = jpkm1; k >= 2; --k) {
            double zemxl = MIND( zmxlm[k + 1] + e3t[I3(i, j, k + 1)], zmxlm[k] );
            zmxlm[k] = zemxl;
            zmxld[k] = zemxl;
        }
    } else {
        for (int k = 2; k <= jpkm1; ++k)
            zmxld[k] = MIND( zmxld[k - 1] + e3t[I3(i, j, k - 1)], zmxlm[k] );
        for (int k = jpkm1; k >= 2; --k)
            zmxlm[k] = MIND( zmxlm[k + 1] + e3t[I3(i, j, k + 1)], zmxlm[k] );
        for (int k = 2; k <= jpkm1; ++k) {
            double zemlm = MIND( zmxld[k], zmxlm[k] );
            double zemlp = sqrt( zmxld[k] * zmxlm[k] );
            zmxlm[k] = zemlm;
            zmxld[k] = zemlp;
        }
    }
    for (int k = 1; k <= jpkm1; ++k) {
        double zsqen = sqrt( en[I3(i, j, k)] );
        double zav = cf.rn_ediff * zmxlm[k] * zsqen;
        avm[I3(i, j, k)] = MAXD( zav, avmb[k - 1] ) * wmask[I3(i, j, k)];
        avt[I3(i, j, k)] = MAXD( zav, avtb_2d[I2(i, j)] * avtb[k - 1] ) * wmask[I3(i, j, k)];
        dissl[I3(i, j, k)] = zsqen / zmxld[k];
    }
    if (cf.nn_pdl == 1) {
        for (int k = 2; k <= jpkm1; ++k)
            avt[I3(i, j, k)] = MAXD( pdlr[k] * avt[I3(i, j, k)], avtb_2d[I2(i, j)] * avtb[k - 1] ) * wmask[I3(i, j, k)];
    }
}
