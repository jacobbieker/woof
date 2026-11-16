// woof/core/kernels/smag2d_tke_faces.cu
//
// smag2d.cu's wrf_tke_rhs (km_opt = 2's forward TKE source), with each
// defor13 / defor23 face deformation evaluated once instead of at every
// point that reads it.  The loader composes smag2d.cu (and its
// glibc_flt32.cuh) ahead of this unit (kernels/__init__.py _EXTRA_HEADERS,
// as smag2d_w_faces.cu and smag2d_uv_faces.cu do), so smag2d.cu keeps every
// byte and the deformations below are smag2d.cu's own wrf_defor13 /
// wrf_defor23, called with the same arguments.
//
// wrf_tke_rhs reads four defor13 and four defor23 faces per mass point (six
// each on the surface level), and every face recomputes its vertical
// metric (two divides), the four-point w averages at two levels and the
// terrain slope from the winds and geopotential: a face was evaluated by up
// to four threads, 3.5 ms per launch at 100x120x120 on an RTX A6000 (5 % of
// the Haleakala LES root step).  wrf_tke_defor_faces stores each face once;
// wrf_tke_rhs_faces is wrf_tke_rhs's statement text with the stored words
// in place of the calls, in the same order (generated from smag2d.cu's text
// by substitution: tests/test_smag2d_tke_faces.py pins that the two bodies
// differ only at the twelve call sites).  Same options as smag2d
// (DIFFUSION_OPTIONS: --fmad=false --ftz=false --prec-div=true
// --prec-sqrt=true): no contraction across what used to be the inlined
// call boundary, so a stored word is the word the call returned; and every
// use squares or sums the returned float itself, so the fused (#if
// !GPUWM_WRF_EXACT && !GPUWM_WRF_EXACT_C_DIFFUSION) statements are the same
// words too.
//
// Buffers: d13 (nz, ny, nx + 1) for j = 0..ny-1, i = 0..nx and d23
// (nz, ny + 1, nx) for j = 0..ny, i = 0..nx-1, level kw = 0..nz-1 (level 0
// is wrf_defor13 / wrf_defor23's own 0.0f, stored; level nz is their 0.0f
// too and is not stored).  wrf_defor13 / wrf_defor23 clamp their own
// horizontal indices under open boundaries before anything is read, so a
// face is stored from the same raw (j, i) a reader forms.

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

extern "C" __global__
void wrf_tke_defor_faces(WRF_SMAG_GRID_ARGS, real* d13, real* d23,
                         int nz, int ny, int nx, int phb3d,
                         int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int kw = blockIdx.z * blockDim.z + threadIdx.z;
    if (i > nx || j > ny || kw >= nz) return;
    WRF_SMAG_MAKE_GRID;
    if (j < ny)
        d13[((SmagIndex)kw * ny + j) * (nx + 1) + i] = wrf_defor13(q, kw, j, i);
    if (i < nx)
        d23[((SmagIndex)kw * (ny + 1) + j) * nx + i] = wrf_defor23(q, kw, j, i);
}

// The stored face at the raw (kw, j, i) wrf_defor13 / wrf_defor23 would
// have been called with; level nz is their 0.0f.
__device__ __forceinline__
real wrf_tke_d13(const WrfSmagGrid& q, const real* d13, int kw, int j, int i)
{
    return kw >= q.nz ? 0.0f : d13[((SmagIndex)kw * q.ny + j) * (q.nx + 1) + i];
}

__device__ __forceinline__
real wrf_tke_d23(const WrfSmagGrid& q, const real* d23, int kw, int j, int i)
{
    return kw >= q.nz ? 0.0f : d23[((SmagIndex)kw * (q.ny + 1) + j) * q.nx + i];
}

extern "C" __global__
void wrf_tke_rhs_faces(WRF_SMAG_GRID_ARGS,
                 const real* d13, const real* d23,
                 const real* thp, const real* thb, int thb3d,
                 const real* tke, const real* bn2,
                 const real* d11a, const real* d22a, const real* d12a,
                 const real* kmh_a, const real* kmv_a, const real* khv_a,
                 const real* mut, const real* c1, const real* c2,
                 const real* ustm, const real* hfx,
                 int use_ustm, int use_hfx,
                 real c_k, real dt, real cd0, real heat_flux_const,
                 int isfflx,
                 real* tend,
                 real* b_shear, real* b_buoy, real* b_diss, real* b_lim,
                 int budget,
                 int nz, int ny, int nx, int phb3d,
                 int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    // WRF tke_rhs excludes the outer mass rows for open lateral boundaries.
    // A nonzero source there advances TKE in cells WRF leaves untouched.
    if ((boundary_x && (i == 0 || i == nx - 1))
        || (boundary_y && (j == 0 || j == ny - 1))) return;
    WRF_SMAG_MAKE_GRID;
    SmagIndex idx = IDX3(k, j, i);
    real chm = c1[k] * mut[(SmagIndex)j * nx + i] + c2[k];
    real kmh = kmh_a[idx], kmv = kmv_a[idx];
    // WRF's tke_rhs accumulates IN PLACE into tke_tend (zeroed by
    // init_zero_tendency, as the dycore zeroes smag_rtke) and the limiter
    // ASSIGNS max(tendency, bound).  Starting from the stored value and
    // storing the limited result keeps WRF's -0.0 where the bound
    // -(c1*mu+c2)*max(0,tke)/dt is -0.0 (tke <= 0): adding -0.0 into a +0.0
    // buffer would turn it into +0.0.
    real t0 = tend[idx];
    real t = t0;

    // --- tke_shear: six squared-deformation production terms.
    real d11 = wrf_d(q, d11a, k, j, i);
    real d22 = wrf_d(q, d22a, k, j, i);
    t += 0.5f * chm * kmh * (d11 * d11);
    t += 0.5f * chm * kmh * (d22 * d22);
    real rdzw_c = wrf_rdzw(q, k, j, i);
    real d33 = 2.0f * (q.w[I3(k + 1, j, i, ny, nx)]
                       - q.w[I3(k, j, i, ny, nx)]) * rdzw_c;
    t += 0.5f * chm * kmv * (d33 * d33);
    real s12a = wrf_d(q, d12a, k, j, i);
    real s12b = wrf_d(q, d12a, k, j + 1, i);
    real s12c = wrf_d(q, d12a, k, j, i + 1);
    real s12d = wrf_d(q, d12a, k, j + 1, i + 1);
    // defor12: avg = 0.25*(four squared corners), then chm*xkmh*avg.
    real avg12 = 0.25f * (s12a * s12a + s12b * s12b
                          + s12c * s12c + s12d * s12d);
    t += chm * kmh * avg12;
    // defor13 (zero at kts and ktf+1, wrf_defor13), then at kts the MARTA
    // u-drag term, then defor23, then the v-drag term: WRF's tke_shear adds
    // them to the running tendency in exactly this order (:6729-6874), and
    // a float32 running sum does not commute.
    real s13a = wrf_tke_d13(q, d13, k + 1, j, i);
    real s13b = wrf_tke_d13(q, d13, k, j, i);
    real s13c = wrf_tke_d13(q, d13, k + 1, j, i + 1);
    real s13d = wrf_tke_d13(q, d13, k, j, i + 1);
    real chm_kmv = chm * kmv;
#if !defined(GPUWM_WRF_EXACT) && !defined(GPUWM_WRF_EXACT_C_DIFFUSION)
    real square13 = __fadd_rn(__fadd_rn(__fadd_rn(
        __fmul_rn(s13a, s13a), __fmul_rn(s13b, s13b)),
        __fmul_rn(s13c, s13c)), __fmul_rn(s13d, s13d));
    t = __fmaf_rn(chm_kmv, __fmul_rn(0.25f, square13), t);
#else
    t += chm_kmv * (0.25f * (s13a * s13a + s13b * s13b
                             + s13c * s13c + s13d * s13d));
#endif
    real usum = 0.0f, vsum = 0.0f, Cd = 0.0f, absU = 0.0f;
    if (k == 0) {
        // MARTA surface drag (u_2/v_2 raw winds, ust at (i,j)):
        // absU = 0.5*sqrt((u(i)+u(i+1))**2 + (v(j)+v(j+1))**2) [+ epsilon].
        usum = q.u[I3S(0, j, wrf_iu(q, i), ny, nx + 1)]
             + q.u[I3S(0, j, wrf_iu(q, i + 1), ny, nx + 1)];
        vsum = q.v[I3S(0, wrf_jv(q, j), i, ny + 1, nx)]
             + q.v[I3S(0, wrf_jv(q, j + 1), i, ny + 1, nx)];
        absU = 0.5f * sqrtf(usum * usum + vsum * vsum);
        if (isfflx == 0) {
            Cd = cd0;
        } else {
            absU += 1.0e-15f;
            real us = use_ustm ? ustm[(SmagIndex)j * nx + i] : 0.0f;
            Cd = (us * us) / (absU * absU);
        }
        // (u(i)+u(i+1))*0.5*Cd*absU*(defor13(i,2)+defor13(i+1,2))*0.5
        real d13sum = wrf_tke_d13(q, d13, 1, j, i) + wrf_tke_d13(q, d13, 1, j, i + 1);
        t += chm * (usum * 0.5f * Cd * absU * d13sum * 0.5f);
    }
    real s23a = wrf_tke_d23(q, d23, k + 1, j, i);
    real s23b = wrf_tke_d23(q, d23, k, j, i);
    real s23c = wrf_tke_d23(q, d23, k + 1, j + 1, i);
    real s23d = wrf_tke_d23(q, d23, k, j + 1, i);
#if !defined(GPUWM_WRF_EXACT) && !defined(GPUWM_WRF_EXACT_C_DIFFUSION)
    real square23 = __fadd_rn(__fadd_rn(__fadd_rn(
        __fmul_rn(s23a, s23a), __fmul_rn(s23b, s23b)),
        __fmul_rn(s23c, s23c)), __fmul_rn(s23d, s23d));
    t = __fmaf_rn(chm_kmv, __fmul_rn(0.25f, square23), t);
#else
    t += chm_kmv * (0.25f * (s23a * s23a + s23b * s23b
                             + s23c * s23c + s23d * s23d));
#endif
    if (k == 0) {
        real d23sum = wrf_tke_d23(q, d23, 1, j, i) + wrf_tke_d23(q, d23, 1, j + 1, i);
        t += chm * (vsum * 0.5f * Cd * absU * d23sum * 0.5f);
    }

    // Budget bookkeeping: each term is the running total's increment, so
    // the four exports sum EXACTLY to the tendency this kernel deposits
    // (t0 is 0 in the forecast, so t - t0 is exact).
    real s_shear = t - t0;

    // --- tke_buoyancy.
    real khv = khv_a[idx];
    if (k >= 1) {
        t -= chm * khv * bn2[idx];
    } else {
        real heat_flux;
        if (isfflx == 0 || isfflx == 2) {
            heat_flux = heat_flux_const;
        } else {
            real vapor = moist ? qv[I3(0, j, i, ny, nx)] : 0.0f;
            real cpm = CP * (1.0f + 0.8f * vapor);
            real hf = use_hfx ? hfx[(SmagIndex)j * nx + i] : 0.0f;
            // grid%rho from phy_prep: 1./alt*(1.+qv), in that order
            // (module_big_step_utilities_em.F:4856).
            real alt0 = q.alt[I3(0, j, i, ny, nx)];
            real rho0 = __fdiv_rn(1.0f, alt0) * (1.0f + vapor);
            heat_flux = (hf / cpm) / rho0;
        }
        real theta_c = wrf_theta_full(q, thp, thb, thb3d, 0, j, i);
        t -= chm * ((khv * bn2[idx])
                    - (G / theta_c) * heat_flux) * 0.5f;
    }

    real s_buoy = t - t0 - s_shear;

    // --- tke_dissip (l_scale from calc_l_scale, computed inline).
    real map = msft[(SmagIndex)j * nx + i];
    real deltas = wrf_tke_deltas(dx, dy, map, rdzw_c);
    real l = wrf_l_scale(tke[idx], bn2[idx], deltas);
    real ce1 = (__fdiv_rn(c_k, 0.10f)) * 0.19f;
    real ce2 = fmaxf(0.0f, 0.93f - ce1);
    real coefc = (k == 0 || k == nz - 1) ? 3.9f
               : (ce1 + ce2 * l / deltas);
    real tketmp = fmaxf(tke[idx], 1.0e-6f);
    // c*coefc*tketmp**1.5/l_scale: ** is the libm power, not tketmp*SQRT.
    t -= chm * coefc * gfk_pow(tketmp, 1.5f) / l;

    real s_diss = t - t0 - s_shear - s_buoy;

    // --- positivity limiter (:6221-6227).
    real t_limited = fmaxf(t, -chm * fmaxf(0.0f, tke[idx]) / dt);
    real s_lim = t_limited - t;
    t = t_limited;
    tend[idx] = t;
    if (budget) {
        b_shear[idx] = s_shear;
        b_buoy[idx] = s_buoy;
        b_diss[idx] = s_diss;
        b_lim[idx] = s_lim;
    }
}

#undef WRF_SMAG_MAKE_GRID
#undef WRF_SMAG_GRID_ARGS
