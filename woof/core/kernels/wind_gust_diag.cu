// woof/core/kernels/wind_gust_diag.cu
//
// WRF nwp_diagnostics wind maxima and a UPP-method surface gust, one column
// per thread, folded into history-interval running maxima.  The Python owner,
// the citations and the ratified divergences are in
// woof/core/wind_gust_diag.py; the NumPy mirror there is the reference this
// kernel is held to bit for bit (tests/test_wind_gust_diag.py).
//
// Members (WRF v4.6.1 phys/module_diag_nwp.F, SUBROUTINE
// diagnostic_output_nwp, every model step under nwp_diagnostics = 1):
//
//   WSPD10MAX  wind_vel = sqrt(u10*u10 + v10*v10); IF (wind_vel .GT.
//              wspd10max) wspd10max = wind_vel
//   U10MAX/    the 10 m components AT the instant WSPD10MAX grew (WRF's
//   V10MAX     clim varstatisticsWIND rule, phys/module_diag_cl.F)
//   W_UP_MAX   DO k = kms, kme: IF (p8w(k) .GT. 40000. .AND. w(k) .GT.
//              w_up_max) w_up_max = w(k)
//   W_DN_MAX   the same with .LT. against w_dn_max
//
// p8w is WRF phy_prep's full-level pressure (dyn_em/
// module_big_step_utilities_em.F): interior fzm(k)*p_phy(k) +
// fzp(k)*p_phy(k-1); bottom w1*p_phy(1) + w2*p_phy(2) with w1 = (z0-z2)/
// (z1-z2) from z_at_w(1) and the two lowest mass heights; top
// exp(w1*log(p_phy(kde-1)) + w2*log(p_phy(kde-2))) from z_at_w(kte) and the
// two highest mass heights.  p_phy = p + pb is woof's state.p (full
// pressure).
//
// GUST: NOAA-EMC UPP CALGUST.f, the RAPR/GFS/FV3R branch: starting from
// SFCWIND = |V10|, for every mass level from the second one up to ONE ABOVE
// the PBL-top level, gust = max(gust, SFCWIND + (|V_k| - SFCWIND) *
// (1 - MIN(0.5, DZ_k/2000))) with DZ_k the mass-level height above the
// surface.  The PBL-top level is the lowest mass level whose height above
// ground is >= PBLH (woof's definition of UPP's LPBL; see the owner).
//
// Bitwise discipline: every FP32 operation is an explicit rounding
// intrinsic, so NVRTC's contraction pass cannot fuse a multiply-add the
// NumPy mirror rounds twice, and sqrt is __fsqrt_rn (IEEE).  The top-level
// p8w uses logf/expf, which are NOT correctly rounded; that value can differ
// from NumPy's by a few ULP, which matters only if the model-top pressure
// sits within those ULP of 40000 Pa (a model top near 7 km).
//
// Layout: C-contiguous (k, j, i); u (nz, ny, nx+1), v (nz, ny+1, nx), w/ph
// (nz+1, ny, nx), p (nz, ny, nx), phb (nz+1, ny, nx) or a flat (nz+1,)
// column (phb3d = 0); fnm/fnp (nz,) with WRF's 1-based fnm(k) at fnm[k-1].

extern "C" {

__device__ __forceinline__ real wg_speed(real a, real b) {
  return __fsqrt_rn(__fadd_rn(__fmul_rn(a, a), __fmul_rn(b, b)));
}

// z_at_w = (phb + ph) / g   (phy_prep; the same expression cal_helicity uses)
__device__ __forceinline__ real wg_z_at_w(
    const real* ph, const real* phb, int phb3d, real g,
    int kw, int jy, int ix, int ny, int nx) {
  const real phv = ph[I3(kw, jy, ix, ny, nx)];
  const real phbv = phb3d ? phb[I3(kw, jy, ix, ny, nx)] : phb[kw];
  return __fdiv_rn(__fadd_rn(phbv, phv), g);
}

__device__ __forceinline__ real wg_z_mass(
    const real* ph, const real* phb, int phb3d, real g,
    int k, int jy, int ix, int ny, int nx) {
  return __fmul_rn(0.5f, __fadd_rn(
      wg_z_at_w(ph, phb, phb3d, g, k, jy, ix, ny, nx),
      wg_z_at_w(ph, phb, phb3d, g, k + 1, jy, ix, ny, nx)));
}

// phy_prep p8w at full level kw (0-based); requires nz >= 2.
__device__ __forceinline__ real wg_p8w(
    const real* p, const real* ph, const real* phb, int phb3d,
    const real* fnm, const real* fnp, real g,
    int kw, int jy, int ix, int nz, int ny, int nx) {
  if (kw == 0) {
    const real z0 = wg_z_at_w(ph, phb, phb3d, g, 0, jy, ix, ny, nx);
    const real z1 = wg_z_mass(ph, phb, phb3d, g, 0, jy, ix, ny, nx);
    const real z2 = wg_z_mass(ph, phb, phb3d, g, 1, jy, ix, ny, nx);
    const real w1 = __fdiv_rn(__fsub_rn(z0, z2), __fsub_rn(z1, z2));
    const real w2 = __fsub_rn(1.0f, w1);
    return __fadd_rn(__fmul_rn(w1, p[I3(0, jy, ix, ny, nx)]),
                     __fmul_rn(w2, p[I3(1, jy, ix, ny, nx)]));
  }
  if (kw == nz) {
    const real z0 = wg_z_at_w(ph, phb, phb3d, g, nz, jy, ix, ny, nx);
    const real z1 = wg_z_mass(ph, phb, phb3d, g, nz - 1, jy, ix, ny, nx);
    const real z2 = wg_z_mass(ph, phb, phb3d, g, nz - 2, jy, ix, ny, nx);
    const real w1 = __fdiv_rn(__fsub_rn(z0, z2), __fsub_rn(z1, z2));
    const real w2 = __fsub_rn(1.0f, w1);
    return expf(__fadd_rn(__fmul_rn(w1, logf(p[I3(nz - 1, jy, ix, ny, nx)])),
                          __fmul_rn(w2, logf(p[I3(nz - 2, jy, ix, ny, nx)]))));
  }
  return __fadd_rn(__fmul_rn(fnm[kw], p[I3(kw, jy, ix, ny, nx)]),
                   __fmul_rn(fnp[kw], p[I3(kw - 1, jy, ix, ny, nx)]));
}

__global__ void wind_maxima_fold(
    const real* __restrict__ u, const real* __restrict__ v,
    const real* __restrict__ w, const real* __restrict__ p,
    const real* __restrict__ ph, const real* __restrict__ phb,
    const int phb3d,
    const real* __restrict__ fnm, const real* __restrict__ fnp,
    const real* __restrict__ u10, const real* __restrict__ v10,
    const real* __restrict__ pblh, const int has_wind10, const int has_pblh,
    const real g,
    real* __restrict__ wspd10max, real* __restrict__ u10max,
    real* __restrict__ v10max,
    real* __restrict__ w_up_max, real* __restrict__ w_dn_max,
    real* __restrict__ gust_max,
    const int nz, const int ny, const int nx) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= nx * ny) return;
  const int jy = col / nx;
  const int ix = col - jy * nx;

  // W_UP_MAX / W_DN_MAX over every full level, k = kms..kme.
  real up = w_up_max[col];
  real dn = w_dn_max[col];
  for (int kw = 0; kw <= nz; ++kw) {
    const real p8w = wg_p8w(p, ph, phb, phb3d, fnm, fnp, g,
                            kw, jy, ix, nz, ny, nx);
    const real wv = w[I3(kw, jy, ix, ny, nx)];
    if (p8w > 40000.0f && wv > up) up = wv;
    if (p8w > 40000.0f && wv < dn) dn = wv;
  }
  w_up_max[col] = up;
  w_dn_max[col] = dn;

  if (!has_wind10) return;

  // WSPD10MAX and the components at its instant.
  const real u10v = u10[col];
  const real v10v = v10[col];
  const real sfc = wg_speed(u10v, v10v);
  if (sfc > wspd10max[col]) {
    wspd10max[col] = sfc;
    u10max[col] = u10v;
    v10max[col] = v10v;
  }

  if (!has_pblh) return;

  // GUST (UPP CALGUST, RAPR/GFS/FV3R branch).
  const real zsfc = wg_z_at_w(ph, phb, phb3d, g, 0, jy, ix, ny, nx);
  const real zpbl = pblh[col];
  int kpbl = nz - 1;
  for (int k = 0; k < nz; ++k) {
    const real dz = __fsub_rn(
        wg_z_mass(ph, phb, phb3d, g, k, jy, ix, ny, nx), zsfc);
    if (dz >= zpbl) { kpbl = k; break; }
  }
  const int ktop = (kpbl + 1 < nz - 1) ? kpbl + 1 : nz - 1;
  real gust = sfc;
  for (int k = 1; k <= ktop; ++k) {
    const real um = __fmul_rn(0.5f, __fadd_rn(
        u[I3S(k, jy, ix, ny, nx + 1)], u[I3S(k, jy, ix + 1, ny, nx + 1)]));
    const real vm = __fmul_rn(0.5f, __fadd_rn(
        v[I3(k, jy, ix, ny + 1, nx)], v[I3(k, jy + 1, ix, ny + 1, nx)]));
    const real wind = wg_speed(um, vm);
    const real dz = __fsub_rn(
        wg_z_mass(ph, phb, phb3d, g, k, jy, ix, ny, nx), zsfc);
    const real q = __fdiv_rn(dz, 2000.0f);
    const real reduce = (q < 0.5f) ? q : 0.5f;
    const real delwind = __fmul_rn(__fsub_rn(wind, sfc),
                                   __fsub_rn(1.0f, reduce));
    const real cand = __fadd_rn(sfc, delwind);
    if (cand > gust) gust = cand;
  }
  if (gust > gust_max[col]) gust_max[col] = gust;
}

}  // extern "C"
