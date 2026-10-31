// woof/core/kernels/smag2d_theta_m.cu
//
// WRF v4.7.1 vertical_diffusion_2's ``use_theta_m = 1`` surface heat flux
// (dyn_em/module_diffusion_em.F:4284-4324, the hflux SELECT CASE), for the
// diff_opt = 2, bl_pbl_physics = 0 path.  The loader composes smag2d.cu
// (and its glibc_flt32.cuh) ahead of this unit (kernels/__init__.py
// _EXTRA_HEADERS), so these entry points read the grid exactly as
// smag2d.cu's own use_theta_m = 0 surface arms do and smag2d.cu keeps every
// byte.  Same options as smag2d (DIFFUSION_OPTIONS: --fmad=false
// --ftz=false --prec-div=true --prec-sqrt=true,
// GPUWM_WRF_EXACT_C_DIFFUSION): each statement is WRF's, left to right, one
// float32 rounding per Fortran operation.  Reference:
// tools/bigstep_wrf471_oracle/thetam_ext_build.py (the hflux block extracted
// unchanged, gfortran -O0 -ffp-contract=off).
//
// ``theta`` is vertical_diffusion_2's ``theta`` argument, phy_prep's full dry
// th_phy (module_first_rk_step_part2.F:1072, module_big_step_utilities_em.F:
// 4830): (t_2 + t0)/(1. + (R_v/R_d)*qv) on the time-t theta_m ``thp`` and
// vapour the grid carries; ``t0`` is woof's base theta ``thb`` ((nz) when
// base3d = 0, else (nz, ny, nx)).  rvovrd is module_model_constants'
// r_v/r_d, the folded word RVOVRD.

// smag2d.cu's argument and grid macros, which it #undefs at its end;
// restated here character for character.
#define WRF_SMAG_GRID_ARGS                                                     \
    const real* u, const real* v, const real* w,                              \
    const real* php, const real* phb, const real* alt, const real* qv,        \
    const real* msft, const real* msfu, const real* msfv,                     \
    const real* fnm, const real* fnp, const real* dn, const real* dnw,         \
    real rdx, real rdy, real dx, real dy, real cf1, real cf2, real cf3,        \
    int moist

#define WRF_SMAG_MAKE_GRID                                                     \
    WrfSmagGrid q = wrf_grid(u, v, w, php, phb, alt, qv, msft, msfu, msfv,    \
                             fnm, fnp, dn, dnw, rdx, rdy, dx, dy,              \
                             cf1, cf2, cf3, nz, ny, nx, phb3d,                 \
                             boundary_x, boundary_y, moist)

__device__ __forceinline__
real wrf_theta_m_surface_th_phy(const WrfSmagGrid& q, const real* thp,
                                const real* thb, int base3d, real vapor,
                                int j, int i, int ny, int nx)
{
    SmagIndex at = I3(0, j, i, ny, nx);
    real t0 = base3d ? thb[at] : thb[0];
    return (thp[at] + t0) / (1.0f + RVOVRD * vapor);
}

// hflux CASE(0,2) (:4286-4302), use_theta_m = 1:
//   cpm = cp * (1. + 0.8 * moist(i,kts,j,P_QV))
//   hfx(i,j) = heat_flux*cpm*rho(i,kts,j)
//   rt_tendf(i,kts,j) = rt_tendf(i,kts,j)
//       - g*heat_flux*(1.+rvovrd*moist(i,kts,j,P_QV))*rho(i,kts,j)/dnw(kts)
//       - g*1.61*theta(i,kts,j)*qfx(i,j)/dnw(kts)
// ``qfx`` is grid%qfx, zero where no surface layer writes it (has_qfx = 0).
extern "C" __global__
void wrf_smag_surface_heat_const_theta_m(
        WRF_SMAG_GRID_ARGS, real heat_flux, real* hfx, int write_hfx,
        const real* qfx, int has_qfx, const real* thp, const real* thb,
        int base3d, real* rth, int nz, int ny, int nx, int phb3d,
        int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    WRF_SMAG_MAKE_GRID;
    real vapor = q.moist ? q.qv[I3(0, j, i, ny, nx)] : 0.0f;
    real rho = wrf_rho(q, 0, j, i);
    if (write_hfx) {
        real cpm = CP * (1.0f + 0.8f * vapor);
        hfx[(SmagIndex)j * nx + i] = heat_flux * cpm * rho;
    }
    real theta = wrf_theta_m_surface_th_phy(q, thp, thb, base3d, vapor,
                                            j, i, ny, nx);
    real flux_q = has_qfx ? qfx[(SmagIndex)j * nx + i] : 0.0f;
    SmagIndex out = I3(0, j, i, ny, nx);
    rth[out] = rth[out]
             - G * heat_flux * (1.0f + RVOVRD * vapor) * rho / dnw[0]
             - G * 1.61f * theta * flux_q / dnw[0];
}

// hflux CASE(1) (:4304-4318), use_theta_m = 1, and the qflux CASE(1,2)
// water-vapour flux smag2d.cu's wrf_smag_surface_scalars adds:
//   cpm = cp * (1. + 0.8 * moist(i,kts,j,P_QV))
//   heat_flux = hfx(i,j)/cpm
//   rt_tendf(i,kts,j) = rt_tendf(i,kts,j)
//       - g*heat_flux*(1.+rvovrd*moist(i,kts,j,P_QV))/dnw(kts)
//       - g*1.61*theta(i,kts,j)*qfx(i,j)/dnw(kts)
extern "C" __global__
void wrf_smag_surface_scalars_theta_m(
        WRF_SMAG_GRID_ARGS, const real* hfx, const real* qfx,
        int apply_heat, int apply_moist, const real* thp, const real* thb,
        int base3d, real* rth, real* rqv, int nz, int ny, int nx, int phb3d,
        int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((!apply_heat && !apply_moist) || i >= nx || j >= ny) return;
    WRF_SMAG_MAKE_GRID;
    SmagIndex h = (SmagIndex)j * nx + i;
    real vapor = moist ? qv[I3(0, j, i, ny, nx)] : 0.0f;
    real cpm = CP * (1.0f + 0.8f * vapor);
    if (apply_heat) {
        real heat_flux = hfx[h] / cpm;
        real theta = wrf_theta_m_surface_th_phy(q, thp, thb, base3d, vapor,
                                                j, i, ny, nx);
        SmagIndex out = I3(0, j, i, ny, nx);
        rth[out] = rth[out]
                 - G * heat_flux * (1.0f + RVOVRD * vapor) / dnw[0]
                 - G * 1.61f * theta * qfx[h] / dnw[0];
    }
    if (apply_moist && moist)
        rqv[I3(0, j, i, ny, nx)] -= G * qfx[h] / dnw[0];
}

#undef WRF_SMAG_MAKE_GRID
#undef WRF_SMAG_GRID_ARGS
