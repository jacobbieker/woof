// WOOF-OPTIN: nemo
// Derived from NEMO 5.0.2 (CeCILL v2.0); see optional/nemo/LICENSE and
// NOTICE.md.  Not part of the default build or of any wheel.
//
// NEMO's equation of state, src/OCE/TRA/eosbn2.F90: eos_insitu_pot_New_t
// (:400-561, the non-stochastic path), rab_3d_t (:1083-1188) and bn2_t
// (:1431-1474), for TEOS-10, EOS-80 (neos = -1, 0: the polynomial, with the
// coefficient table eos.py prepends as EOS###/ALP###/BET### macros over c[])
// and S-EOS (neos = 1).  NEMO computes in double; each expression is NEMO's,
// in its order, with no contraction (--fmad=false; Metal's soft binary64
// never contracts).  Fields are (jpk, jpj, jpi) C order, NEMO's memory order.

extern "C" __global__
void nemo_eos_insitu_pot(const double* __restrict__ c, const double* __restrict__ tem,
                         const double* __restrict__ sal, const double* __restrict__ gdept,
                         const double* __restrict__ tmask, double* __restrict__ prd,
                         double* __restrict__ prhop, int neos, double rn_T0, double rn_S0,
                         double rn_a0, double rn_b0, double rn_lambda1, double rn_lambda2,
                         double rn_mu1, double rn_mu2, double rn_nu, double rho0,
                         double r1_rho0, int jpk, int n2)
{
    int c_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_ >= n2) return;
    for (int jk = 0; jk < jpk - 1; ++jk) {
        size_t o = (size_t)jk * n2 + c_;
        double ztm = tmask[o];
        if (neos != 1) {
            double zh = gdept[o] * r1_Z0;
            double zt = tem[o] * r1_T0;
            double zs = sqrt(fabs(sal[o] + rdeltaS) * r1_S0);
            double zn3 = EOS013*zt
                + EOS103*zs+EOS003;
            double zn2 = (EOS022*zt
                + EOS112*zs+EOS012)*zt
                + (EOS202*zs+EOS102)*zs+EOS002;
            double zn1 = (((EOS041*zt
                + EOS131*zs+EOS031)*zt
                + (EOS221*zs+EOS121)*zs+EOS021)*zt
                + ((EOS311*zs+EOS211)*zs+EOS111)*zs+EOS011)*zt
                + (((EOS401*zs+EOS301)*zs+EOS201)*zs+EOS101)*zs+EOS001;
            double zn0 = (((((EOS060*zt
                + EOS150*zs+EOS050)*zt
                + (EOS240*zs+EOS140)*zs+EOS040)*zt
                + ((EOS330*zs+EOS230)*zs+EOS130)*zs+EOS030)*zt
                + (((EOS420*zs+EOS320)*zs+EOS220)*zs+EOS120)*zs+EOS020)*zt
                + ((((EOS510*zs+EOS410)*zs+EOS310)*zs+EOS210)*zs+EOS110)*zs+EOS010)*zt
                + (((((EOS600*zs+EOS500)*zs+EOS400)*zs+EOS300)*zs+EOS200)*zs+EOS100)*zs+EOS000;
            double zn = ( ( zn3 * zh + zn2 ) * zh + zn1 ) * zh + zn0;
            prhop[o] = zn0 * ztm;
            prd[o] = ( zn * r1_rho0 - 1.0 ) * ztm;
        } else {
            double zt = tem[o] - rn_T0;
            double zs = sal[o] - rn_S0;
            double zh = gdept[o];
            double zn = - rn_a0 * ( 1.0 + 0.5*rn_lambda1*zt ) * zt
                + rn_b0 * ( 1.0 - 0.5*rn_lambda2*zs ) * zs
                - rn_nu * zt * zs;
            prhop[o] = ( rho0 + zn ) * ztm;
            zn = zn - ( rn_a0 * rn_mu1 * zt + rn_b0 * rn_mu2 * zs ) * zh;
            prd[o] = zn * r1_rho0 * ztm;
        }
    }
}

extern "C" __global__
void nemo_rab(const double* __restrict__ c, const double* __restrict__ tem,
              const double* __restrict__ sal, const double* __restrict__ gdept,
              const double* __restrict__ tmask, double* __restrict__ alpha,
              double* __restrict__ beta, int neos, double rn_T0, double rn_S0,
              double rn_a0, double rn_b0, double rn_lambda1, double rn_lambda2,
              double rn_mu1, double rn_mu2, double rn_nu, double r1_rho0, int jpk, int n2)
{
    int c_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_ >= n2) return;
    for (int jk = 0; jk < jpk - 1; ++jk) {
        size_t o = (size_t)jk * n2 + c_;
        double ztm = tmask[o];
        if (neos != 1) {
            double zh = gdept[o] * r1_Z0;
            double zt = tem[o] * r1_T0;
            double zs = sqrt(fabs(sal[o] + rdeltaS) * r1_S0);
            double zn3 = ALP003;
            double zn2 = ALP012*zt + ALP102*zs+ALP002;
            double zn1 = ((ALP031*zt
                + ALP121*zs+ALP021)*zt
                + (ALP211*zs+ALP111)*zs+ALP011)*zt
                + ((ALP301*zs+ALP201)*zs+ALP101)*zs+ALP001;
            double zn0 = ((((ALP050*zt
                + ALP140*zs+ALP040)*zt
                + (ALP230*zs+ALP130)*zs+ALP030)*zt
                + ((ALP320*zs+ALP220)*zs+ALP120)*zs+ALP020)*zt
                + (((ALP410*zs+ALP310)*zs+ALP210)*zs+ALP110)*zs+ALP010)*zt
                + ((((ALP500*zs+ALP400)*zs+ALP300)*zs+ALP200)*zs+ALP100)*zs+ALP000;
            double zn = ( ( zn3 * zh + zn2 ) * zh + zn1 ) * zh + zn0;
            alpha[o] = zn * r1_rho0 * ztm;
            zn3 = BET003;
            zn2 = BET012*zt + BET102*zs+BET002;
            zn1 = ((BET031*zt
                + BET121*zs+BET021)*zt
                + (BET211*zs+BET111)*zs+BET011)*zt
                + ((BET301*zs+BET201)*zs+BET101)*zs+BET001;
            zn0 = ((((BET050*zt
                + BET140*zs+BET040)*zt
                + (BET230*zs+BET130)*zs+BET030)*zt
                + ((BET320*zs+BET220)*zs+BET120)*zs+BET020)*zt
                + (((BET410*zs+BET310)*zs+BET210)*zs+BET110)*zs+BET010)*zt
                + ((((BET500*zs+BET400)*zs+BET300)*zs+BET200)*zs+BET100)*zs+BET000;
            zn = ( ( zn3 * zh + zn2 ) * zh + zn1 ) * zh + zn0;
            beta[o] = zn / zs * r1_rho0 * ztm;
        } else {
            double zt = tem[o] - rn_T0;
            double zs = sal[o] - rn_S0;
            double zh = gdept[o];
            double zn = rn_a0 * ( 1.0 + rn_lambda1*zt + rn_mu1*zh ) + rn_nu*zs;
            alpha[o] = zn * r1_rho0 * ztm;
            zn = rn_b0 * ( 1.0 - rn_lambda2*zs - rn_mu2*zh ) - rn_nu*zt;
            beta[o] = zn * r1_rho0 * ztm;
        }
    }
}

extern "C" __global__
void nemo_bn2(const double* __restrict__ tem, const double* __restrict__ sal,
              const double* __restrict__ alpha, const double* __restrict__ beta,
              const double* __restrict__ gdept, const double* __restrict__ gdepw,
              const double* __restrict__ e3w, const double* __restrict__ wmask,
              double* __restrict__ pn2, double grav, int jpk, int n2)
{
    int c_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_ >= n2) return;
    for (int jk = 1; jk < jpk - 1; ++jk) {
        size_t o = (size_t)jk * n2 + c_, u = (size_t)(jk - 1) * n2 + c_;
        double zrw = ( gdepw[o] - gdept[o] )
                   / ( gdept[u] - gdept[o] );
        double zaw = alpha[o] * (1. - zrw) + alpha[u] * zrw;
        double zbw = beta[o] * (1. - zrw) + beta[u] * zrw;
        pn2[o] = grav * (  zaw * ( tem[u] - tem[o] )
                         - zbw * ( sal[u] - sal[o] )  )
               / e3w[o] * wmask[o];
    }
}
