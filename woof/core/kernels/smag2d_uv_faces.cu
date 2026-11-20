// woof/core/kernels/smag2d_uv_faces.cu
//
// smag2d.cu's wrf_smag_hd_u / wrf_smag_hd_v, with the magnitude of each
// horizontal stress coefficient (rho K at the mass points for tau11 and
// tau22, rhoavg kavg at the cell corners for tau12) evaluated once instead
// of at every point that reads it.  The loader composes smag2d.cu (and its
// glibc_flt32.cuh) ahead of this unit (kernels/__init__.py _EXTRA_HEADERS, as
// smag2d_w_faces.cu does), so smag2d.cu keeps every byte and the
// stresses below are smag2d.cu's own wrf_tau11 / wrf_tau22 / wrf_tau12
// with the product of their first two factors loaded.
//
// wrf_smag_hd_u reads two tau11 and two tau12 stresses for its horizontal
// divergence and eight of each for the two vertical averages, and every
// stress recomputes its density (a divide) and, for tau12, four-point
// density and K averages: a mass-point stress was evaluated by up to ten
// u threads and a corner stress by up to ten, 2.8 ms + 1.0 ms (hd_u + hd_v)
// of a 49x120x150 step on an RTX A6000.  wrf_smag_uv_coef_faces stores each
// coefficient once; wrf_smag_hd_u_faces / wrf_smag_hd_v_faces are the hd_u /
// hd_v statements with each stress spelled -stored * deformation, in the
// same order.
//
// Why the coefficient and not the stress: the default (fast) numerics drop
// --fmad=false (woof.numerics.effective_options), so NVRTC contracts a
// stress product into the add or subtract that consumes it.  A stored
// stress is a rounded product that cannot be contracted, so its sums round
// differently (measured: about two thirds of the u tendency words on a 3 km
// forecast step).  The products rho * K and rhoavg * kavg feed only the
// stress product, never an add, and rho itself is spelled with explicit _rn
// operators, so the stored word is the word every build computes inline
// ((-a) * b is -(a * b) exactly).  The reader keeps the stress text's
// leading negation, -stored * d: the compiler hoists that sign out of the
// products before it contracts, and a stored negative coefficient changed
// the shape of tau_a - tau_b (measured on the same step).
//
// Defined where wrf_smag_hd_u / wrf_smag_hd_v take their plain (#else)
// statements: GPUWM_WRF_EXACT or GPUWM_WRF_EXACT_C_DIFFUSION.  Buffers: cm
// (rho K, nz, ny, nx) at the mass points (the stress helpers clamp or wrap every
// mass index through wrf_iy / wrf_ix, so the readers below clamp the same
// way before the load); c12 (rhoavg kavg, nz, ny + 1, nx + 1) at the corners j = 0..ny,
// i = 0..nx, every corner index a reader forms, stored from the same raw
// (j, i) wrf_tau12 would have been called with.

#if defined(GPUWM_WRF_EXACT) || defined(GPUWM_WRF_EXACT_C_DIFFUSION)

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
void wrf_smag_uv_coef_faces(WRF_SMAG_GRID_ARGS, const real* km,
                            real* cm, real* c12,
                            int nz, int ny, int nx, int phb3d,
                            int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i > nx || j > ny || k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    // wrf_tau11 / wrf_tau22: -wrf_rho * wrf_k * wrf_d, left to right; the
    // first product's magnitude.
    if (i < nx && j < ny)
        cm[I3(k, j, i, ny, nx)] = wrf_rho(q, k, j, i) * wrf_k(q, km, k, j, i);
    // wrf_tau12: -rhoavg * kavg * wrf_d, left to right, its averages
    // spelled as there; the first product's magnitude.
    real rhoavg = 0.25f * (wrf_rho(q, k, j, i)
                            + wrf_rho(q, k, j, i - 1)
                            + wrf_rho(q, k, j - 1, i - 1)
                            + wrf_rho(q, k, j - 1, i));
    real kavg = 0.25f * (wrf_k(q, km, k, j, i)
                          + wrf_k(q, km, k, j, i - 1)
                          + wrf_k(q, km, k, j - 1, i - 1)
                          + wrf_k(q, km, k, j - 1, i));
    c12[((SmagIndex)k * (ny + 1) + j) * (nx + 1) + i] = rhoavg * kavg;
}

// smag2d.cu's wrf_tau11 / wrf_tau22 take any mass index and clamp or wrap
// it inside wrf_rho / wrf_k / wrf_d; the stored coefficient is read at the
// same clamped point, under the stress text's leading negation.
__device__ __forceinline__
real wrf_smag_uv_t11(const WrfSmagGrid& q, const real* cm, const real* d11,
                     int k, int j, int i)
{
    return -cm[I3(k, wrf_iy(q, j), wrf_ix(q, i), q.ny, q.nx)]
           * wrf_d(q, d11, k, j, i);
}

__device__ __forceinline__
real wrf_smag_uv_t22(const WrfSmagGrid& q, const real* cm, const real* d22,
                     int k, int j, int i)
{
    return -cm[I3(k, wrf_iy(q, j), wrf_ix(q, i), q.ny, q.nx)]
           * wrf_d(q, d22, k, j, i);
}

// Corner stress at the raw (j, i) the reader forms (0..ny, 0..nx): minus
// the stored product times wrf_tau12's deformation read.
__device__ __forceinline__
real wrf_smag_uv_t12(const WrfSmagGrid& q, const real* c12, const real* d12,
                     int k, int j, int i)
{
    return -c12[((SmagIndex)k * (q.ny + 1) + j) * (q.nx + 1) + i]
           * wrf_d(q, d12, k, j, i);
}

// smag2d.cu's wrf_tau11_uavg / wrf_tau12_uavg / wrf_tau12_vavg /
// wrf_tau22_vavg (their plain #else statements) on the stored stresses, in
// the same summation order.
__device__ __forceinline__
real wrf_smag_uv_t11_uavg(const WrfSmagGrid& q, const real* cm,
                          const real* d11, int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    return 0.5f * (q.fnm[kw] * (wrf_smag_uv_t11(q, cm, d11, kw, j, i - 1)
                                 + wrf_smag_uv_t11(q, cm, d11, kw, j, i))
                  + q.fnp[kw] * (wrf_smag_uv_t11(q, cm, d11, kw - 1, j, i - 1)
                                 + wrf_smag_uv_t11(q, cm, d11, kw - 1, j, i)));
}

__device__ __forceinline__
real wrf_smag_uv_t12_uavg(const WrfSmagGrid& q, const real* c12,
                          const real* d12, int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    return 0.5f * (q.fnm[kw] * (wrf_smag_uv_t12(q, c12, d12, kw, j + 1, i)
                                 + wrf_smag_uv_t12(q, c12, d12, kw, j, i))
                  + q.fnp[kw] * (wrf_smag_uv_t12(q, c12, d12, kw - 1, j + 1, i)
                                 + wrf_smag_uv_t12(q, c12, d12, kw - 1, j, i)));
}

__device__ __forceinline__
real wrf_smag_uv_t12_vavg(const WrfSmagGrid& q, const real* c12,
                          const real* d12, int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    return 0.5f * (q.fnm[kw] * (wrf_smag_uv_t12(q, c12, d12, kw, j, i + 1)
                                 + wrf_smag_uv_t12(q, c12, d12, kw, j, i))
                  + q.fnp[kw] * (wrf_smag_uv_t12(q, c12, d12, kw - 1, j, i + 1)
                                 + wrf_smag_uv_t12(q, c12, d12, kw - 1, j, i)));
}

__device__ __forceinline__
real wrf_smag_uv_t22_vavg(const WrfSmagGrid& q, const real* cm,
                          const real* d22, int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    return 0.5f * (q.fnm[kw] * (wrf_smag_uv_t22(q, cm, d22, kw, j - 1, i)
                                 + wrf_smag_uv_t22(q, cm, d22, kw, j, i))
                  + q.fnp[kw] * (wrf_smag_uv_t22(q, cm, d22, kw - 1, j - 1, i)
                                 + wrf_smag_uv_t22(q, cm, d22, kw - 1, j, i)));
}

extern "C" __global__
void wrf_smag_hd_u_faces(WRF_SMAG_GRID_ARGS, const real* cm, const real* c12,
                         const real* d11, const real* d12, real* tend,
                         int nz, int ny, int nx, int phb3d,
                         int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx + 1 || j >= ny || k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    int ic = wrf_ix(q, i), im = wrf_ix(q, i - 1);
    int iu = wrf_iu(q, i), jj = wrf_iy(q, j);
    real msf = msfu[(SmagIndex)jj * (nx + 1) + iu];
    real tmpdz = 0.5f * (1.0f / wrf_rdzw(q, k, j, ic)
                          + 1.0f / wrf_rdzw(q, k, j, im));
    real zx_u = 0.5f * (wrf_zx(q, k, j, i) + wrf_zx(q, k + 1, j, i));
    real zy_u = 0.125f * (
        wrf_zy(q, k, j, im) + wrf_zy(q, k, j, ic)
      + wrf_zy(q, k, j + 1, im) + wrf_zy(q, k, j + 1, ic)
      + wrf_zy(q, k + 1, j, im) + wrf_zy(q, k + 1, j, ic)
      + wrf_zy(q, k + 1, j + 1, im) + wrf_zy(q, k + 1, j + 1, ic));
    real divh = msf * rdx * (wrf_smag_uv_t11(q, cm, d11, k, j, ic)
                              - wrf_smag_uv_t11(q, cm, d11, k, j, im))
              + msf * rdy * (wrf_smag_uv_t12(q, c12, d12, k, j + 1, i)
                              - wrf_smag_uv_t12(q, c12, d12, k, j, i));
#ifdef GPUWM_WRF_EXACT_C_DIFFUSION
    real first = __fdiv_rn(msf * zx_u * (wrf_smag_uv_t11_uavg(q, cm, d11, k + 1, j, i) - wrf_smag_uv_t11_uavg(q, cm, d11, k, j, i)), tmpdz);
    real second = __fdiv_rn(msf * zy_u * (wrf_smag_uv_t12_uavg(q, c12, d12, k + 1, j, i) - wrf_smag_uv_t12_uavg(q, c12, d12, k, j, i)), tmpdz);
    tend[I3S(k, j, i, ny, nx + 1)] += __fdiv_rn(G * tmpdz, dnw[k]) * ((divh - first) - second);
#else
    real divz = msf * zx_u * (wrf_smag_uv_t11_uavg(q, cm, d11, k + 1, j, i)
                              - wrf_smag_uv_t11_uavg(q, cm, d11, k, j, i)) / tmpdz
              + msf * zy_u * (wrf_smag_uv_t12_uavg(q, c12, d12, k + 1, j, i)
                              - wrf_smag_uv_t12_uavg(q, c12, d12, k, j, i)) / tmpdz;
    tend[I3S(k, j, i, ny, nx + 1)] += G * tmpdz / dnw[k] * (divh - divz);
#endif
}

extern "C" __global__
void wrf_smag_hd_v_faces(WRF_SMAG_GRID_ARGS, const real* cm, const real* c12,
                         const real* d22, const real* d12, real* tend,
                         int nz, int ny, int nx, int phb3d,
                         int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny + 1 || k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    int jc = wrf_iy(q, j), jm = wrf_iy(q, j - 1);
    int jv = wrf_jv(q, j), ii = wrf_ix(q, i);
    real msf = msfv[(SmagIndex)jv * nx + ii];
    real tmpdz = 0.5f * (1.0f / wrf_rdzw(q, k, jc, i)
                          + 1.0f / wrf_rdzw(q, k, jm, i));
#ifdef GPUWM_WRF_EXACT_C_DIFFUSION
    real zx_v=0.125f*(wrf_zx(q,k,jc,i)+wrf_zx(q,k,jc,i+1)+wrf_zx(q,k,jm,i)+wrf_zx(q,k,jm,i+1)+wrf_zx(q,k+1,jc,i)+wrf_zx(q,k+1,jc,i+1)+wrf_zx(q,k+1,jm,i)+wrf_zx(q,k+1,jm,i+1));
#else
    real zx_v = 0.125f * (
        wrf_zx(q, k, jm, i) + wrf_zx(q, k, jm, i + 1)
      + wrf_zx(q, k, jc, i) + wrf_zx(q, k, jc, i + 1)
      + wrf_zx(q, k + 1, jm, i) + wrf_zx(q, k + 1, jm, i + 1)
      + wrf_zx(q, k + 1, jc, i) + wrf_zx(q, k + 1, jc, i + 1));
#endif
    real zy_v = 0.5f * (wrf_zy(q, k, j, i) + wrf_zy(q, k + 1, j, i));
    real divh = msf * rdy * (wrf_smag_uv_t22(q, cm, d22, k, jc, i)
                              - wrf_smag_uv_t22(q, cm, d22, k, jm, i))
              + msf * rdx * (wrf_smag_uv_t12(q, c12, d12, k, j, i + 1)
                              - wrf_smag_uv_t12(q, c12, d12, k, j, i));
#ifdef GPUWM_WRF_EXACT_C_DIFFUSION
    real first = __fdiv_rn(msf * zx_v * (wrf_smag_uv_t12_vavg(q, c12, d12, k + 1, j, i) - wrf_smag_uv_t12_vavg(q, c12, d12, k, j, i)), tmpdz);
    real second = __fdiv_rn(msf * zy_v * (wrf_smag_uv_t22_vavg(q, cm, d22, k + 1, j, i) - wrf_smag_uv_t22_vavg(q, cm, d22, k, j, i)), tmpdz);
    tend[I3S(k, j, i, ny + 1, nx)] += __fdiv_rn(G * tmpdz, dnw[k]) * ((divh - first) - second);
#else
    real divz = msf * zx_v * (wrf_smag_uv_t12_vavg(q, c12, d12, k + 1, j, i)
                              - wrf_smag_uv_t12_vavg(q, c12, d12, k, j, i)) / tmpdz
              + msf * zy_v * (wrf_smag_uv_t22_vavg(q, cm, d22, k + 1, j, i)
                              - wrf_smag_uv_t22_vavg(q, cm, d22, k, j, i)) / tmpdz;
    tend[I3S(k, j, i, ny + 1, nx)] += G * tmpdz / dnw[k] * (divh - divz);
#endif
}

#undef WRF_SMAG_MAKE_GRID
#undef WRF_SMAG_GRID_ARGS

#endif
