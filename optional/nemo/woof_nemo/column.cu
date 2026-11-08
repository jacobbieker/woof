// WOOF-OPTIN: nemo
// Derived from NEMO 5.0.2 (CeCILL v2.0); see optional/nemo/LICENSE and
// NOTICE.md.  Not part of the default build or of any wheel.
//
// The pieces of NEMO's MLF step (src/OCE/stpmlf.F90) that join the ported
// physics into a coupled tracer column:
//   nemo_sbc_rcv     sbc_ssm's nn_fsbc = 1 lines (SBC/sbcssm.F90) with
//                    eos_pt_from_ct (TRA/eosbn2.F90), then sbc_cpl_rcv's
//                    ocean-only mapping of WRF's six fields (SBC/sbccpl.F90)
//   nemo_drg_cd0     drg_init's masked drag coefficient (ZDF/zdfdrg.F90)
//   nemo_drg_cdu     zdf_drg_lin / zdf_drg_nonlin, the bottom -Cd*|U|
//   nemo_tra_sbc     tra_sbc for a linear free surface (TRA/trasbc.F90)
//   nemo_tra_atf     tra_atf_fix_lf, the Asselin filter (TRA/traatf_qco.F90)
//   nemo_cpl_snd_sst sbc_cpl_snd's O_SSTSST, 'oce only' (SBC/sbccpl.F90)
// Double, NEMO's expressions in NEMO's order.  3-D fields are (jpk, jpj, jpi)
// C order with a one-cell halo; the WRF planes are (jpj-2, jpi-2) float32,
// the interior.  One thread per interior column.

#define I3(i, j, k) ((size_t)((k) - 1) * n2 + (size_t)((j) - 1) * nx + ((i) - 1))
#define I2(i, j) ((size_t)((j) - 1) * nx + ((i) - 1))
#define IW(i, j) ((size_t)((j) - 2) * (nx - 2) + ((i) - 2))
#define COLUMN_THREAD                                         \
    int t = blockIdx.x * blockDim.x + threadIdx.x;            \
    int ni = nx - 2, nj = ny - 2;                             \
    if (t >= ni * nj) return;                                 \
    int i = t % ni + 2, j = t / ni + 2;                       \
    size_t n2 = (size_t)nx * ny;                              \
    (void)n2;

// eos_pt_from_ct_t (eosbn2.F90 :1493-1548)
__device__ static double nemo_pt_from_ct(double ctmp, double psal, double ztm)
{
    double zdeltaS = 5.0;
    double z1_S0 = 0.875 / 35.16504;
    double z1_T0 = 1.0 / 40.0;
    double zt = ctmp * z1_T0;
    double zs = sqrt( fabs( psal + zdeltaS ) * z1_S0 );
    double zn = ((((-2.1385727895e-01*zt
                  - 2.7674419971e-01*zs+1.0728094330)*zt
                  + (2.6366564313*zs+3.3546960647)*zs-7.8012209473)*zt
                  + ((1.8835586562*zs+7.3949191679)*zs-3.3937395875)*zs-5.6414948432)*zt
                  + (((3.5737370589*zs-1.5512427389e+01)*zs+2.4625741105e+01)*zs
                     +1.9912291000e+01)*zs-3.2191146312e+01)*zt
                  + ((((5.7153204649e-01*zs-3.0943149543)*zs+9.3052495181)*zs
                     -9.4528934807)*zs+3.1066408996)*zs-4.3504021262e-01;
    double zd = (2.0035003456*zt
                 -3.4570358592e-01*zs+5.6471810638)*zt
                 + (1.5393993508*zs-6.9394762624)*zs+1.2750522650e+01;
    return ( zt / z1_T0 + zn / zd ) * ztm;
}

// frcv indices, wrf_cpl.AtmosphereToOcean's field order
#define F_OEMP 0
#define F_QSROCE 1
#define F_QNSOCE 2
#define F_OTX1 3
#define F_OTY1 4
#define F_TAUM 5
#define F_W10M 6

extern "C" __global__
void nemo_sbc_rcv(const double* __restrict__ tn, const double* __restrict__ sn,
                  const double* __restrict__ tmask, int use_ct, int received,
                  const float* __restrict__ evap_precip, const float* __restrict__ surf_net_solar,
                  const float* __restrict__ surf_net_non_solar, const float* __restrict__ taux,
                  const float* __restrict__ tauy, const float* __restrict__ taumod,
                  double* __restrict__ frcv, double* __restrict__ sst_m, double* __restrict__ sss_m,
                  double* __restrict__ utau, double* __restrict__ vtau, double* __restrict__ taum,
                  double* __restrict__ wndm, double* __restrict__ emp, double* __restrict__ qns,
                  double* __restrict__ qsr, double rcp, int nx, int ny)
{
    COLUMN_THREAD
    size_t c = I2(i, j), w = IW(i, j);
    // sbc_ssm, nn_fsbc = 1 (sbcssm.F90 :70-76): ts(:,:,1,:,Kmm)
    double zt = tn[I3(i, j, 1)], zs = sn[I3(i, j, 1)];
    sst_m[c] = use_ct ? nemo_pt_from_ct(zt, zs, tmask[I3(i, j, 1)]) : zt;
    sss_m[c] = zs;
    // sbc_cpl_rcv: cpl_rcv fills frcv only at a coupling time; the coupler
    // hands NEMO the REAL(4) values exactly
#define FR(f) frcv[(size_t)(f) * n2 + c]
    if (received) {
        FR(F_OEMP) = (double)evap_precip[w];
        FR(F_QSROCE) = (double)surf_net_solar[w];
        FR(F_QNSOCE) = (double)surf_net_non_solar[w];
        FR(F_OTX1) = (double)taux[w];
        FR(F_OTY1) = (double)tauy[w];
        FR(F_TAUM) = (double)taumod[w];
        // stress received on the local grid: no geo2oce, no rot_rep
        // O_TauMod received and new (:140-144)
        FR(F_TAUM) = fmax( 0.0, FR(F_TAUM) );
        // w10m not received: from the new stress module (:149-158), zrhoa
        // and zcdrag REAL(wp) set from default-real literals
        double zrhoa = (double)1.22f, zcdrag = (double)1.5e-3f;
        double zcoef = 1.0 / ( zrhoa * zcdrag );
        FR(F_W10M) = sqrt( FR(F_TAUM) * zcoef );
    }
    // MOD(kt-1, k_fsbc) == 0 with nn_fsbc = 1, no mixed coupling (:181-186)
    utau[c] = FR(F_OTX1);
    vtau[c] = FR(F_OTY1);
    taum[c] = FR(F_TAUM);
    wndm[c] = FR(F_W10M);
    // ocean-only heat and freshwater fluxes (:367-431): 'oce only' E-P, no
    // calving, runoff, icebergs or ice shelves; nn_components = 0 is not
    // jp_iam_oce, so the heat content of the mass flux leaves at SST
    double zemp = FR(F_OEMP);
    emp[c] = zemp;
    double zqns = FR(F_QNSOCE);
    zqns = zqns - zemp * sst_m[c] * rcp;
    qns[c] = zqns;
    qsr[c] = FR(F_QSROCE);
#undef FR
}

// drg_init (zdfdrg.F90 :439-531), bottom, no boost: pCd0 = rn_Cd0 * ssmask
// for the linear and non-linear drags; the log layer on a linear free
// surface folds (vkarmn / LOG(z/z0))**2, bounded, into it once.  mode: 0
// OFF, 1 lin, 2 non_lin, 3 loglayer.
extern "C" __global__
void nemo_drg_cd0(const double* __restrict__ ssmask, const double* __restrict__ e3t_0,
                  const int* __restrict__ mbkt, int mode, double rn_Cd0, double rn_Cdmax,
                  double rn_z0, double vkarmn, double* __restrict__ pCd0, int nx, int ny)
{
    COLUMN_THREAD
    size_t c = I2(i, j);
    double zmsk_boost = 1.0 * ssmask[c];
    if (mode == 0) {
        pCd0[c] = 0.0;
    } else if (mode == 3) {
        double zzz = 0.5 * e3t_0[I3(i, j, mbkt[c])];
        double zcd = vkarmn / glibc_log( zzz / rn_z0 );
        zcd = zcd * zcd;
        double zb = rn_Cd0 >= zcd ? rn_Cd0 : zcd;          // MAX( rn_Cd0 , zcd )
        pCd0[c] = zmsk_boost * ( zb <= rn_Cdmax ? zb : rn_Cdmax );
    } else {
        pCd0[c] = rn_Cd0 * zmsk_boost;
    }
}

// zdf_drg_lin (:198-224) for modes 0/1 at init; zdf_drg_nonlin's standard
// Cd (:181-188) every step for modes 2/3 (l_log_not_linssh is false)
extern "C" __global__
void nemo_drg_cdu(const double* __restrict__ pCd0, const double* __restrict__ uu,
                  const double* __restrict__ vv, const int* __restrict__ mbkt, int mode,
                  double rn_Uc0, double pke0, double* __restrict__ pCdU, int nx, int ny)
{
    COLUMN_THREAD
    size_t c = I2(i, j);
    if (mode == 0) {
        pCdU[c] = 0.0;
    } else if (mode == 1) {
        pCdU[c] = - pCd0[c] * rn_Uc0;
    } else {
        int imk = mbkt[c];
        double zut = uu[I3(i, j, imk)] + uu[I3(i - 1, j, imk)];
        double zvt = vv[I3(i, j, imk)] + vv[I3(i, j - 1, imk)];
        pCdU[c] = - pCd0[c] * sqrt( 0.25 * ( zut*zut + zvt*zvt ) + pke0 );
    }
}

// tra_sbc (trasbc.F90 :64-170), lk_linssh, no runoff, no restart
extern "C" __global__
void nemo_tra_sbc(double* __restrict__ qns, double* __restrict__ qsr,
                  const double* __restrict__ emp, const double* __restrict__ sfx,
                  double* __restrict__ sbc_tsc, double* __restrict__ sbc_tsc_b,
                  const double* __restrict__ tn, const double* __restrict__ sn,
                  const double* __restrict__ e3t, double* __restrict__ trhs,
                  double* __restrict__ srhs, int ln_traqsr, int first,
                  double r1_rho0_rcp, double r1_rho0, int nx, int ny)
{
    COLUMN_THREAD
    size_t c = I2(i, j);
    if (!ln_traqsr) {                         // no penetration: all heat in qns
        qns[c] = qns[c] + qsr[c];
        qsr[c] = 0.0;
    }
    double zfact;
    if (first) {                              // kt == nit000, no restart: Euler
        zfact = 1.0;
        sbc_tsc[c] = 0.0; sbc_tsc[n2 + c] = 0.0;
        sbc_tsc_b[c] = 0.0; sbc_tsc_b[n2 + c] = 0.0;
    } else {
        zfact = 0.5;
        sbc_tsc_b[c] = sbc_tsc[c]; sbc_tsc_b[n2 + c] = sbc_tsc[n2 + c];
    }
    sbc_tsc[c] = r1_rho0_rcp * qns[c];
    sbc_tsc[n2 + c] = r1_rho0 * sfx[c];
    sbc_tsc[c] = sbc_tsc[c] + r1_rho0 * emp[c] * tn[I3(i, j, 1)];
    sbc_tsc[n2 + c] = sbc_tsc[n2 + c] + r1_rho0 * emp[c] * sn[I3(i, j, 1)];
    trhs[I3(i, j, 1)] = trhs[I3(i, j, 1)] + zfact * ( sbc_tsc_b[c] + sbc_tsc[c] )
                      / e3t[I3(i, j, 1)];
    srhs[I3(i, j, 1)] = srhs[I3(i, j, 1)] + zfact * ( sbc_tsc_b[n2 + c] + sbc_tsc[n2 + c] )
                      / e3t[I3(i, j, 1)];
}

// tra_atf_fix_lf (traatf_qco.F90 :186-219), levels 1..jpkm1, both tracers
extern "C" __global__
void nemo_tra_atf(const double* __restrict__ tb, const double* __restrict__ sb,
                  double* __restrict__ tn, double* __restrict__ sn,
                  const double* __restrict__ ta, const double* __restrict__ sa,
                  double rn_atfp, int nx, int ny, int jpk)
{
    COLUMN_THREAD
    for (int jn = 0; jn < 2; ++jn) {
        const double* pb = jn ? sb : tb;
        const double* pa = jn ? sa : ta;
        double* pn = jn ? sn : tn;
        for (int k = 1; k <= jpk - 1; ++k) {
            double ztn = pn[I3(i, j, k)];
            double ztd = pa[I3(i, j, k)] - 2.0 * ztn + pb[I3(i, j, k)];
            pn[I3(i, j, k)] = ztn + rn_atfp * ztd;
        }
    }
}

// sbc_cpl_snd (sbccpl.F90 :2228-2236), nn_components = 0, 'oce only':
// potential temperature + rt0; the coupler hands WRF the REAL(4) value
extern "C" __global__
void nemo_cpl_snd_sst(const double* __restrict__ tn, const double* __restrict__ sn,
                      const double* __restrict__ tmask, int use_ct, double rt0,
                      float* __restrict__ sst, int nx, int ny)
{
    COLUMN_THREAD
    double zt = tn[I3(i, j, 1)];
    double z = use_ct ? nemo_pt_from_ct(zt, sn[I3(i, j, 1)], tmask[I3(i, j, 1)]) : zt;
    z = z + rt0;
    sst[IW(i, j)] = __double2float_rn(z);
}
