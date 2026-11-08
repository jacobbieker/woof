// WOOF-OPTIN: nemo
// Derived from NEMO 5.0.2 (CeCILL v2.0); see optional/nemo/LICENSE and
// NOTICE.md.  Not part of the default build or of any wheel.
//
// NEMO's flux-corrected transport advection of a tracer,
// src/OCE/TRA/traadv_fct.F90 (tra_adv_fct with fct_up1_1stp, nonosc and
// interp_4th_cpt) for leapfrog (no key_RK3), a linear free surface, no
// ice-shelf cavities and no adaptive-implicit vertical advection, with
// 2nd- or 4th-order centred horizontal and 2nd-order centred or 4th-order
// compact vertical anti-diffusive fluxes.  Double, NEMO's expressions in
// NEMO's order.  Fields are (jpk, jpj, jpi) C order with NEMO 5.0's
// two-cell halo (H = 2); every loop keeps NEMO's index range, and the one
// lbc_lnk (on the upstream-corrected fluxes and the upstream guess) is
// NEMO's single-process closed-boundary fill: the halo set to zero.  One
// thread per column (i, j) of the whole array; 1-based indices as in NEMO.

#define I3(i, j, k) ((size_t)((k) - 1) * n2 + (size_t)((j) - 1) * nx + ((i) - 1))
#define I2(i, j) ((size_t)((j) - 1) * nx + ((i) - 1))
// gfortran -O0 lowers MAX(a, b) / MIN(a, b) to x86 maxsd/minsd with b as
// the source operand, which is returned when the two compare equal (so
// MAX(-0., 0.) is +0.): the signed zeros below the sea floor depend on it.
#define MAXD(a, b) ((a) > (b) ? (a) : (b))
#define MIND(a, b) ((a) < (b) ? (a) : (b))
#define ADV_KMAX 128
#define COLUMN                                                    \
    int t = blockIdx.x * blockDim.x + threadIdx.x;                \
    if (t >= nx * ny) return;                                     \
    int i = t % nx + 1, j = t / nx + 1;                           \
    size_t n2 = (size_t)nx * ny;                                  \
    int jpkm1 = jpk - 1;                                          \
    int is0 = H + 1, ie0 = nx - H, js0 = H + 1, je0 = ny - H;     \
    (void)is0; (void)ie0; (void)js0; (void)je0; (void)jpkm1;
#define IN(i, j, lo_i, hi_i, lo_j, hi_j) \
    ((i) >= is0 - (lo_i) && (i) <= ie0 + (hi_i) && (j) >= js0 - (lo_j) && (j) <= je0 + (hi_j))
#define RING(i, j) ((i) <= H || (i) > nx - H || (j) <= H || (j) > ny - H)

// fct_up1_1stp (:378-468): upstream fluxes.  Fu, Fv on DO_3D(1,0,1,0),
// Fw on DO_3D(1,1,1,1) for jk = 2..jpkm1 and its surface value; level jpk
// of every flux is zero (:124-127).
extern "C" __global__
void nemo_fct_upstream(const double* __restrict__ pU, const double* __restrict__ pV,
                       const double* __restrict__ pW, const double* __restrict__ tb,
                       const double* __restrict__ wmask, double* __restrict__ Fu,
                       double* __restrict__ Fv, double* __restrict__ Fw,
                       int nx, int ny, int jpk, int H)
{
    COLUMN
    if (IN(i, j, 1, 0, 1, 0))
        for (int k = 1; k <= jpkm1; ++k) {
            size_t o = I3(i, j, k);
            Fu[o] = ( MAXD( pU[o], 0.0 ) * tb[o] + MIND( pU[o], 0.0 ) * tb[I3(i + 1, j, k)] );
            Fv[o] = ( MAXD( pV[o], 0.0 ) * tb[o] + MIND( pV[o], 0.0 ) * tb[I3(i, j + 1, k)] );
        }
    if (IN(i, j, 1, 1, 1, 1)) {
        for (int k = 2; k <= jpkm1; ++k) {
            size_t o = I3(i, j, k);
            Fw[o] = ( MAXD( pW[o], 0.0 ) * tb[o] + MIND( pW[o], 0.0 ) * tb[I3(i, j, k - 1)] )
                  * wmask[o];
        }
        Fw[I3(i, j, 1)] = pW[I3(i, j, 1)] * tb[I3(i, j, 1)];   // lk_linssh, no cavities
    }
    Fu[I3(i, j, jpk)] = 0.0; Fv[I3(i, j, jpk)] = 0.0; Fw[I3(i, j, jpk)] = 0.0;
}

// fct_up1_1stp (:425-437): the upstream trend into the RHS and the upstream
// guess on the interior; elsewhere the guess is what lbc_lnk makes it, 0.
extern "C" __global__
void nemo_fct_up1(const double* __restrict__ Fu, const double* __restrict__ Fv,
                  const double* __restrict__ Fw, const double* __restrict__ tb,
                  const double* __restrict__ e3t, const double* __restrict__ tmask,
                  const double* __restrict__ r1_e1e2t, double* __restrict__ rhs,
                  double* __restrict__ up1, double pDt, int nx, int ny, int jpk, int H)
{
    COLUMN
    bool interior = IN(i, j, 0, 0, 0, 0);
    for (int k = 1; k <= jpk; ++k) {
        size_t o = I3(i, j, k);
        if (!interior || k == jpk) { up1[o] = 0.0; continue; }
        double ztra = - (  ( Fu[o] - Fu[I3(i - 1, j, k)] )
                         + ( Fv[o] - Fv[I3(i, j - 1, k)] )
                         + ( Fw[o] - Fw[I3(i, j, k + 1)] ) ) * r1_e1e2t[I2(i, j)];
        rhs[o] = rhs[o] + ztra / e3t[o] * tmask[o];
        up1[o] = ( e3t[o] * tb[o] + pDt * ztra ) / e3t[o] * tmask[o];
    }
}

// The anti-diffusive fluxes (:183-279): horizontal on DO_3D(1,0,1,0), 2nd
// or 4th order (ztu/ztv inline: the same words on DO_2D(2,1,2,1));
// vertical on DO_3D(1,1,1,1) for jk = 2..jpkm1, 2nd order or the compact
// 4th-order interp_4th_cpt (:1002-1082), with the surface value zero
// (:282-286).  Then lbc_lnk's closed fill of Fu and Fv: zero on the halo.
__device__ static double fct_ztu(const double* tn, const double* umask, int i, int j, int k,
                                 int nx, size_t n2)
{
    return ( tn[I3(i + 1, j, k)] - tn[I3(i, j, k)] ) * umask[I3(i, j, k)];
}
__device__ static double fct_ztv(const double* tn, const double* vmask, int i, int j, int k,
                                 int nx, size_t n2)
{
    return ( tn[I3(i, j + 1, k)] - tn[I3(i, j, k)] ) * vmask[I3(i, j, k)];
}

extern "C" __global__
void nemo_fct_antidiffusive(const double* __restrict__ pU, const double* __restrict__ pV,
                            const double* __restrict__ pW, const double* __restrict__ tn,
                            const double* __restrict__ umask, const double* __restrict__ vmask,
                            const double* __restrict__ wmask, const int* __restrict__ mbkt,
                            const int* __restrict__ mikt, double* __restrict__ Fu,
                            double* __restrict__ Fv, double* __restrict__ Fw, int kn_fct_h,
                            int kn_fct_v, int nx, int ny, int jpk, int H)
{
    COLUMN
    const double r1_6 = 1.0 / 6.0;
    if (IN(i, j, 1, 0, 1, 0))
        for (int k = 1; k <= jpkm1; ++k) {
            size_t o = I3(i, j, k);
            if (kn_fct_h == 2) {
                Fu[o] = 0.5 * pU[o] * ( tn[o] + tn[I3(i + 1, j, k)] ) - Fu[o];
                Fv[o] = 0.5 * pV[o] * ( tn[o] + tn[I3(i, j + 1, k)] ) - Fv[o];
            } else {
                double zC2t_u = tn[o] + tn[I3(i + 1, j, k)];
                double zC2t_v = tn[o] + tn[I3(i, j + 1, k)];
                double zC4t_u = zC2t_u + r1_6 * ( fct_ztu(tn, umask, i - 1, j, k, nx, n2)
                                                - fct_ztu(tn, umask, i + 1, j, k, nx, n2) );
                double zC4t_v = zC2t_v + r1_6 * ( fct_ztv(tn, vmask, i, j - 1, k, nx, n2)
                                                - fct_ztv(tn, vmask, i, j + 1, k, nx, n2) );
                Fu[o] = 0.5 * pU[o] * zC4t_u - Fu[o];
                Fv[o] = 0.5 * pV[o] * zC4t_v - Fv[o];
            }
        }
    if (IN(i, j, 1, 1, 1, 1)) {
        if (kn_fct_v == 2) {
            for (int k = 2; k <= jpkm1; ++k) {
                size_t o = I3(i, j, k);
                Fw[o] = (  pW[o] * 0.5 * ( tn[o] + tn[I3(i, j, k - 1)] ) - Fw[o]  ) * wmask[o];
            }
        } else if (jpk < ADV_KMAX) {
            // interp_4th_cpt: the compact interpolation of T at w-points
            double zwd[ADV_KMAX], zwi[ADV_KMAX], zws[ADV_KMAX], zwrm[ADV_KMAX], zwt[ADV_KMAX];
            double ztw[ADV_KMAX];
            for (int k = 3; k <= jpkm1; ++k) {
                size_t o = I3(i, j, k);
                zwd[k] = 3.0 * wmask[o] + 1.0;
                zwi[k] = wmask[o];
                zws[k] = wmask[o];
                zwrm[k] = 3.0 * wmask[o] * ( tn[o] + tn[I3(i, j, k - 1)] );
            }
            int ikt = mikt[I2(i, j)] + 1;
            int ikb = MAXD(mbkt[I2(i, j)], 2);
            zwd[ikt] = 1.0; zwi[ikt] = 0.0; zws[ikt] = 0.0;
            zwrm[ikt] = 0.5 * ( tn[I3(i, j, ikt - 1)] + tn[I3(i, j, ikt)] );
            zwd[ikb] = 1.0; zwi[ikb] = 0.0; zws[ikb] = 0.0;
            zwrm[ikb] = 0.5 * ( tn[I3(i, j, ikb - 1)] + tn[I3(i, j, ikb)] );
            zwt[2] = zwd[2];
            for (int k = 3; k <= jpkm1; ++k)
                zwt[k] = zwd[k] - zwi[k] * zws[k - 1] / zwt[k - 1];
            ztw[2] = zwrm[2];
            for (int k = 3; k <= jpkm1; ++k)
                ztw[k] = zwrm[k] - zwi[k] / zwt[k - 1] * ztw[k - 1];
            ztw[jpkm1] = ztw[jpkm1] / zwt[jpkm1];
            for (int k = jpk - 2; k >= 2; --k)
                ztw[k] = ( ztw[k] - zws[k] * ztw[k + 1] ) / zwt[k];
            for (int k = 2; k <= jpkm1; ++k) {
                size_t o = I3(i, j, k);
                Fw[o] = ( pW[o] * ztw[k] - Fw[o] ) * wmask[o];
            }
        }
        Fw[I3(i, j, 1)] = 0.0;
    }
    if (RING(i, j))                       // lbc_lnk: closed boundary, zland = 0
        for (int k = 1; k <= jpk; ++k) { Fu[I3(i, j, k)] = 0.0; Fv[I3(i, j, k)] = 0.0; }
}

// nonosc (:741-936), the limiter coefficients on DO_2D(1,1,1,1) for every
// level.  Each level's zbetup/zbetdo reads the fluxes before any of them
// is limited (NEMO limits a level only after computing its coefficients,
// and reads only unlimited fluxes there), so the levels are independent.
// The bounds use NEMO's three-slot window: the level above is the level
// itself at jk = 1 and the level below is the level itself at jpkm1.
__device__ static void fct_bounds(const double* tb, const double* up1, const double* tmask,
                                  int i, int j, int k, int nx, size_t n2, double& up, double& dn)
{
    const double zbig = 1.7976931348623157e308;   // HUGE(1._wp)
    size_t o = I3(i, j, k);
    bool wet = tmask[o] == 1.0;
    up = wet ? MAXD( tb[o], up1[o] ) : -zbig;
    dn = wet ? MIND( tb[o], up1[o] ) :  zbig;
}

extern "C" __global__
void nemo_fct_beta(const double* __restrict__ tb, const double* __restrict__ up1,
                   const double* __restrict__ tmask, const double* __restrict__ Fu,
                   const double* __restrict__ Fv, const double* __restrict__ Fw,
                   const double* __restrict__ e1e2t, const double* __restrict__ e3t,
                   double* __restrict__ betup, double* __restrict__ betdo, double p2dt,
                   int nx, int ny, int jpk, int H)
{
    COLUMN
    if (!IN(i, j, 1, 1, 1, 1)) return;
    const double zbig = 1.7976931348623157e308;
    double z1_Dt = 1.0 / p2dt;
    for (int jk = 1; jk <= jpkm1; ++jk) {
        int km = jk == 1 ? 1 : jk - 1;
        int kp = jk == jpkm1 ? jpkm1 : jk + 1;
        double u0, d0, u1, d1, u2, d2, u3, d3, u4, d4, u5, d5, u6, d6;
        fct_bounds(tb, up1, tmask, i, j, jk, nx, n2, u0, d0);
        fct_bounds(tb, up1, tmask, i - 1, j, jk, nx, n2, u1, d1);
        fct_bounds(tb, up1, tmask, i + 1, j, jk, nx, n2, u2, d2);
        fct_bounds(tb, up1, tmask, i, j - 1, jk, nx, n2, u3, d3);
        fct_bounds(tb, up1, tmask, i, j + 1, jk, nx, n2, u4, d4);
        fct_bounds(tb, up1, tmask, i, j, km, nx, n2, u5, d5);
        fct_bounds(tb, up1, tmask, i, j, kp, nx, n2, u6, d6);
        double zup = MAXD(MAXD(MAXD(MAXD(MAXD(MAXD(u0, u1), u2), u3), u4), u5), u6);
        double zdo = MIND(MIND(MIND(MIND(MIND(MIND(d0, d1), d2), d3), d4), d5), d6);
        size_t o = I3(i, j, jk);
        double zpos = MAXD( 0.0, Fu[I3(i - 1, j, jk)] ) - MIND( 0.0, Fu[o] )
                    + MAXD( 0.0, Fv[I3(i, j - 1, jk)] ) - MIND( 0.0, Fv[o] )
                    + MAXD( 0.0, Fw[I3(i, j, jk + 1)] ) - MIND( 0.0, Fw[o] );
        double zneg = MAXD( 0.0, Fu[o] ) - MIND( 0.0, Fu[I3(i - 1, j, jk)] )
                    + MAXD( 0.0, Fv[o] ) - MIND( 0.0, Fv[I3(i, j - 1, jk)] )
                    + MAXD( 0.0, Fw[o] ) - MIND( 0.0, Fw[I3(i, j, jk + 1)] );
        double zbt = e1e2t[I2(i, j)] * e3t[o] * z1_Dt;
        betup[o] = ( zup != -zbig && zpos != 0.0 ) ? ( zup - up1[o] ) / zpos * zbt : zbig;
        betdo[o] = ( zdo !=  zbig && zneg != 0.0 ) ? ( up1[o] - zdo ) / zneg * zbt : zbig;
    }
}

// nonosc's flux limiting: Fu, Fv on DO_2D(1,0,1,0) at every level, Fw on
// the interior below the surface; then the corrected trend into the RHS on
// the interior (:312-318).  Two launches: limit, then trend.
extern "C" __global__
void nemo_fct_limit(const double* __restrict__ betup, const double* __restrict__ betdo,
                    double* __restrict__ Fu, double* __restrict__ Fv, double* __restrict__ Fw,
                    int nx, int ny, int jpk, int H)
{
    COLUMN
    if (IN(i, j, 1, 0, 1, 0))
        for (int k = 1; k <= jpkm1; ++k) {
            size_t o = I3(i, j, k);
            double zcoef = Fu[o] > 0.0
                ? MIND( MIND( 1.0, betdo[o] ), betup[I3(i + 1, j, k)] )
                : MIND( MIND( 1.0, betup[o] ), betdo[I3(i + 1, j, k)] );
            Fu[o] = Fu[o] * zcoef;
            zcoef = Fv[o] > 0.0
                ? MIND( MIND( 1.0, betdo[o] ), betup[I3(i, j + 1, k)] )
                : MIND( MIND( 1.0, betup[o] ), betdo[I3(i, j + 1, k)] );
            Fv[o] = Fv[o] * zcoef;
        }
    if (IN(i, j, 0, 0, 0, 0))
        for (int k = 2; k <= jpkm1; ++k) {
            size_t o = I3(i, j, k);
            double zcoef = Fw[o] > 0.0
                ? MIND( MIND( 1.0, betdo[o] ), betup[I3(i, j, k - 1)] )
                : MIND( MIND( 1.0, betup[o] ), betdo[I3(i, j, k - 1)] );
            Fw[o] = Fw[o] * zcoef;
        }
}

extern "C" __global__
void nemo_fct_trend(const double* __restrict__ Fu, const double* __restrict__ Fv,
                    const double* __restrict__ Fw, const double* __restrict__ e3t,
                    const double* __restrict__ tmask, const double* __restrict__ r1_e1e2t,
                    double* __restrict__ rhs, int nx, int ny, int jpk, int H)
{
    COLUMN
    if (!IN(i, j, 0, 0, 0, 0)) return;
    for (int k = 1; k <= jpkm1; ++k) {
        size_t o = I3(i, j, k);
        double ztra = - (  ( Fu[o] - Fu[I3(i - 1, j, k)] )
                         + ( Fv[o] - Fv[I3(i, j - 1, k)] )
                         + ( Fw[o] - Fw[I3(i, j, k + 1)] ) ) * r1_e1e2t[I2(i, j)];
        rhs[o] = rhs[o] + ztra / e3t[o] * tmask[o];
    }
}
