// woof/core/kernels/theta_m.cu
//
// WRF v4.7.1 ``use_theta_m = 1``: the moist potential temperature
// prognostic, theta_m = theta*(1 + Rv/Rd*qv) (Registry.EM_COMMON:2864, the
// "thm" record of grid%t_2).  These are the statements of every
// ``use_theta_m`` branch that converts between the dycore's theta_m and the
// dry theta the physics sees, transcribed in Fortran's left-to-right order
// with one round-to-nearest float32 operation per Fortran operation (the
// unit compiles --fmad=false --ftz=false, so nothing contracts and nothing
// flushes).  The reference is gfortran -O0 -ffp-contract=off over the
// unmodified routine bodies (tools/bigstep_wrf471_oracle/thetam_build.py).
//
// WRF writes ``T0 + t`` with t the perturbation from T0 = 300.  woof's
// perturbation is from the base theta ``thb``; strict arithmetic keeps thb at
// exactly 300 (woof.core.state.adopt_wrf_theta_carrier), where ``thb + thp``
// IS WRF's ``t + T0`` word for word, and every "- T0" below is "- thb".  In
// the default arithmetic thb is the base profile and the same statements
// hold with T0 read as the column's base theta.  ``thb`` is (nz) for a flat
// base (base3d = 0) or (nz, ny, nx) (base3d = 1), as calc_p_alpha reads it.
// (R_v/R_d) is a constant expression gfortran folds once, correctly rounded:
// the float32 word RVOVRD (woof.core.constants).
//
// Layout: mass fields are (nz, ny, nx), ``ncol = ny*nx``, one thread per
// mass point.

__device__ __forceinline__ float tm_base(const float* __restrict__ thb,
                                         int base3d, size_t idx, size_t ncol)
{
    return base3d ? thb[idx] : thb[idx / ncol];
}

// module_initialize_real.F:4923-4931 (module_initialize_ideal.F:1726, the
// same statement): ``t_2 = ( t_2 + T0 ) * (1. + (R_v/R_d) * qv) - T0``, the
// conversion real.exe/ideal.exe make at the very end of initialization, so
// the initial condition and every boundary built from it carry theta_m.
extern "C" __global__
void theta_m_from_dry(float* __restrict__ thp, const float* __restrict__ qv,
                      const float* __restrict__ thb, int base3d,
                      int nz, int ncol)
{
    const size_t n = (size_t)nz * (size_t)ncol;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float t0 = tm_base(thb, base3d, idx, (size_t)ncol);
    const float qvf = __fadd_rn(1.0f, __fmul_rn(RVOVRD, qv[idx]));
    thp[idx] = __fsub_rn(__fmul_rn(__fadd_rn(thp[idx], t0), qvf), t0);
}

// phy_prep (module_big_step_utilities_em.F:4830-4852), moist_physics_prep_em
// (:5505-5512) and the diagnostics driver's update_phys_fields
// (module_diagnostics_driver.F:1252-1258): the dry theta the physics sees,
//   th_phy      = (t + t0)/(1. + R_v/R_d*qv)
//   th_phy_m_t0 = th_phy - t0
// ``dry`` receives th_phy - thb (the dry perturbation; may alias ``thp``).
extern "C" __global__
void theta_dry_from_moist(const float* __restrict__ thp,
                          const float* __restrict__ qv,
                          const float* __restrict__ thb, int base3d,
                          int nz, int ncol, float* dry)
{
    const size_t n = (size_t)nz * (size_t)ncol;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float t0 = tm_base(thb, base3d, idx, (size_t)ncol);
    const float qvf = __fadd_rn(1.0f, __fmul_rn(RVOVRD, qv[idx]));
    const float th_phy = __fdiv_rn(__fadd_rn(thp[idx], t0), qvf);
    dry[idx] = __fsub_rn(th_phy, t0);
}

// conv_t_tendf_to_moist (module_big_step_utilities_em.F:6674-6707), called
// once per step after update_phy_ten (module_first_rk_step_part2.F:1005):
//   t_tendf = (1. + (R_v/R_d) * moist_old)*t_tendf
//           + (R_v/R_d)*(t_1+T0)/(1.+R_v/R_d*moist_old)*moist_tend
// ``rt`` is the held dry-theta physics tendency, ``rqv`` the held coupled
// vapour tendency, ``qv_old`` the step-start vapour (initialize_moist_old)
// and ``t1`` the step-start theta_m.  ``rqv_msf`` = 1 divides the vapour
// term by msft first: the default arithmetic hands the dycore an rt already
// divided by the mass-point map factor (woof.core.physics, rk_addtend_dry's
// division taken early), while strict arithmetic keeps WRF's undivided words
// and divides the folded sum later.  ``out`` may alias ``rt``.
extern "C" __global__
void conv_t_tendf_to_moist(const float* __restrict__ rt,
                           const float* __restrict__ rqv,
                           const float* __restrict__ qv_old,
                           const float* __restrict__ t1,
                           const float* __restrict__ thb, int base3d,
                           const float* __restrict__ msft, int rqv_msf,
                           int nz, int ncol, float* out)
{
    const size_t n = (size_t)nz * (size_t)ncol;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float t0 = tm_base(thb, base3d, idx, (size_t)ncol);
    const float qvf = __fadd_rn(1.0f, __fmul_rn(RVOVRD, qv_old[idx]));
    float moist_tend = rqv[idx];
    if (rqv_msf) moist_tend = __fdiv_rn(moist_tend, msft[idx % (size_t)ncol]);
    const float term1 = __fmul_rn(qvf, rt[idx]);
    const float term2 = __fmul_rn(
        __fdiv_rn(__fmul_rn(RVOVRD, __fadd_rn(t1[idx], t0)), qvf), moist_tend);
    out[idx] = __fadd_rn(term1, term2);
}

// moist_physics_finish_em (module_big_step_utilities_em.F:5680-5746), the
// ``use_theta_m = 1`` branch under no_mp_heating = 0:
//   mpten = th_phy - h_diabatic ; clamped to +/- mp_tend_lim*dt
//   qvten = qv - qv_diabatic
//   t_new = h_diabatic*(1. + (R_v/R_d)*qv_diabatic) + mpten*(1. + (R_v/R_d)*qv)
//         + (R_v/R_d)*qvten*th_phy - T0
//   th_phy_m_t0 = (t_new+T0)/(1.+(R_v/R_d)*qv) - T0
//   h_diabatic  = ( mpten*(1. + (R_v/R_d)*qv) + (R_v/R_d)*qvten*th_phy ) / dt
// ``th_phy`` is the scheme's post-microphysics full dry theta, ``h_diabatic``
// on entry the pre-microphysics full dry theta moist_physics_prep_em parked
// there, ``qv_diabatic`` the pre-microphysics vapour.  ``lim`` is the REAL
// product mp_tend_lim*dt WRF forms.  ``thp`` receives t_new (theta_m'),
// ``dry`` th_phy_m_t0.
extern "C" __global__
void moist_physics_finish_theta_m(const float* __restrict__ th_phy,
                                  float* __restrict__ h_diabatic,
                                  const float* __restrict__ qv_diabatic,
                                  const float* __restrict__ qv,
                                  const float* __restrict__ thb, int base3d,
                                  float lim, float dt, int nz, int ncol,
                                  float* __restrict__ thp,
                                  float* __restrict__ dry)
{
    const size_t n = (size_t)nz * (size_t)ncol;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float t0 = tm_base(thb, base3d, idx, (size_t)ncol);
    const float th = th_phy[idx];
    const float h = h_diabatic[idx];
    const float qvd = qv_diabatic[idx];
    const float q = qv[idx];
    float mpten = __fsub_rn(th, h);
    const float qvten = __fsub_rn(q, qvd);
    mpten = (lim < mpten) ? lim : mpten;          // min(mp_tend_lim*dt, mpten)
    mpten = (-lim > mpten) ? -lim : mpten;        // max(-mp_tend_lim*dt, mpten)
    const float qvf_old = __fadd_rn(1.0f, __fmul_rn(RVOVRD, qvd));
    const float qvf = __fadd_rn(1.0f, __fmul_rn(RVOVRD, q));
    const float a = __fmul_rn(h, qvf_old);
    const float b = __fmul_rn(mpten, qvf);
    const float c = __fmul_rn(__fmul_rn(RVOVRD, qvten), th);
    const float t_new = __fsub_rn(__fadd_rn(__fadd_rn(a, b), c), t0);
    thp[idx] = t_new;
    dry[idx] = __fsub_rn(__fdiv_rn(__fadd_rn(t_new, t0), qvf), t0);
    h_diabatic[idx] = __fdiv_rn(__fadd_rn(b, c), dt);
}
