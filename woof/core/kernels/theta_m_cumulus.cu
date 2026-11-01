// woof/core/kernels/theta_m_cumulus.cu
//
// WRF v4.7.1 phy_prep_part2's ``use_theta_m = 1`` advective-forcing block
// (dyn_em/module_big_step_utilities_em.F:5299-5333), for the cumulus
// schemes that read RTHFTEN (GFSCHEME, G3SCHEME, NTIEDTKESCHEME; GDSCHEME is
// not ported).  Under use_theta_m = 1 the dycore's advective theta forcing
// is the transport of theta_m; phy_prep_part2 hands the scheme the dry-theta
// forcing:
//
//   th_phy  = (t_new + t0) / (1. + (R_v/R_d) * qv)
//   rthften = th_phy/(t_new+t0) * (rthften - (R_v/R_d) * th_phy * rqvften)
//
// after decoupling both forcings by (c1(k)*MUT + c2(k)) with c1 = c1h,
// c2 = c2h (:5299-5321) when ``decouple`` is set (strict Grell-Freitas keeps
// WRF's coupled pair until here; every other route exports them already
// decoupled).  Fortran's left-to-right order, one round-to-nearest float32
// operation per Fortran operation (--fmad=false --ftz=false); (R_v/R_d) is
// the folded constant RVOVRD.  ``t0`` is woof's base theta ``thb`` (exactly
// 300 in strict arithmetic, woof.core.state.adopt_wrf_theta_carrier), (nz)
// for a flat base (base3d = 0) or (nz, ny, nx).  Reference:
// tools/bigstep_wrf471_oracle/thetam_ext_build.py, gfortran -O0
// -ffp-contract=off over the unmodified routine.
//
// Layout: mass fields (nz, ny, nx), ncol = ny*nx, one thread per point.

extern "C" __global__
void phy_prep_part2_rthften_theta_m(float* __restrict__ rthften,
                                    float* __restrict__ rqvften,
                                    const float* __restrict__ t_new,
                                    const float* __restrict__ qv,
                                    const float* __restrict__ thb, int base3d,
                                    const float* __restrict__ c1h,
                                    const float* __restrict__ c2h,
                                    const float* __restrict__ mut,
                                    int decouple, int nz, int ncol)
{
    const size_t n = (size_t)nz * (size_t)ncol;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const size_t k = idx / (size_t)ncol;
    const size_t col = idx % (size_t)ncol;
    const float t0 = base3d ? thb[idx] : thb[k];
    float rqv = rqvften[idx];
    float rth = rthften[idx];
    if (decouple) {
        // RQVFTEN = RQVFTEN/(c1(k)*MUT(I,J)+c2(k)), then RTHFTEN likewise.
        const float mass = __fadd_rn(__fmul_rn(c1h[k], mut[col]), c2h[k]);
        rqv = __fdiv_rn(rqv, mass);
        rth = __fdiv_rn(rth, mass);
        rqvften[idx] = rqv;
    }
    const float tm = __fadd_rn(t_new[idx], t0);
    const float th_phy = __fdiv_rn(tm, __fadd_rn(1.0f, __fmul_rn(RVOVRD, qv[idx])));
    const float ratio = __fdiv_rn(th_phy, tm);
    const float moist = __fmul_rn(__fmul_rn(RVOVRD, th_phy), rqv);
    rthften[idx] = __fmul_rn(ratio, __fsub_rn(rth, moist));
}
