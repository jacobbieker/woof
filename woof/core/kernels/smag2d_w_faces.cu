// woof/core/kernels/smag2d_w_faces.cu
//
// smag2d.cu's wrf_smag_hd_w, with each tau13/tau23 face stress evaluated
// once instead of at every point that reads it.  The loader composes
// smag2d.cu (and its glibc_flt32.cuh) ahead of this unit (kernels/__init__.py
// _EXTRA_HEADERS), so smag2d.cu keeps every byte and the stresses below are
// smag2d.cu's own wrf_tau13 / wrf_tau23, called with the same arguments.
//
// wrf_smag_hd_w reads twelve face stresses per w point (two each for the
// horizontal divergence, four for each of the two vertical averages) and
// every face stress recomputes its deformation from the winds (two metric
// divides, four-point w averages at two levels and the terrain slope), so
// one face's deformation was evaluated up to twelve times: 13.1 ms of a
// 76 ms step at 200x200x50 on an RTX A6000.  wrf_smag_w_defor_faces stores
// each defor13 / defor23 face once; wrf_smag_hd_w_faces is wrf_smag_hd_w's
// statement, in the same order, with each stress spelled as wrf_tau13 /
// wrf_tau23 spell it (zero outside 0 < k < nz, then -rhoavg * kavg *
// deformation) and the stored word in place of the deformation call.
//
// The density and K averages stay at the reader.  Strict numerics compile
// with DIFFUSION_OPTIONS (--fmad=false), but the default (fast) numerics
// drop --fmad=false and NVRTC contracts across the inlined calls: a stored
// stress, or a stored rhoavg * kavg coefficient, moved about a quarter of
// a 3 km forecast step's w words (the inlined averages contract with their
// neighbours').  A deformation ends in an add and only ever feeds the
// stress product, so its stored word is the word every inlined call
// produced (checked against wrf_defor13 / wrf_defor23 on that step).
//
// Defined where wrf_smag_hd_w takes its plain (#else) statement:
// GPUWM_WRF_EXACT or GPUWM_WRF_EXACT_C_DIFFUSION.  Buffers: dx13 (nz, ny,
// nx + 1) for i = 0..nx, dy23 (nz, ny + 1, nx) for j = 0..ny, levels
// k = 1..nz-1; level 0 and level nz are wrf_tau13 / wrf_tau23's own 0.0f
// and are neither stored nor read.

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
void wrf_smag_w_defor_faces(WRF_SMAG_GRID_ARGS, real* dx13, real* dy23,
                            int nz, int ny, int nx, int phb3d,
                            int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int kw = blockIdx.z * blockDim.z + threadIdx.z;
    if (i > nx || j > ny || kw <= 0 || kw >= nz) return;
    WRF_SMAG_MAKE_GRID;
    if (j < ny)
        dx13[((SmagIndex)kw * ny + j) * (nx + 1) + i] = wrf_defor13(q, kw, j, i);
    if (i < nx)
        dy23[((SmagIndex)kw * (ny + 1) + j) * nx + i] = wrf_defor23(q, kw, j, i);
}

// smag2d.cu's wrf_tau13 / wrf_tau23, character for character but for the
// stored deformation word in place of the wrf_defor13 / wrf_defor23 call.
__device__ __forceinline__
real wrf_smag_w_tx(const WrfSmagGrid& q, const real* km, const real* dx13,
                   int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    real rhoavg = 0.5f * (q.fnm[kw] * (wrf_rho(q, kw, j, i - 1)
                                        + wrf_rho(q, kw, j, i))
                           + q.fnp[kw] * (wrf_rho(q, kw - 1, j, i - 1)
                                        + wrf_rho(q, kw - 1, j, i)));
    real kavg = 0.5f * (q.fnm[kw] * (wrf_k(q, km, kw, j, i - 1)
                                      + wrf_k(q, km, kw, j, i))
                         + q.fnp[kw] * (wrf_k(q, km, kw - 1, j, i - 1)
                                      + wrf_k(q, km, kw - 1, j, i)));
    return -rhoavg * kavg * dx13[((SmagIndex)kw * q.ny + j) * (q.nx + 1) + i];
}

__device__ __forceinline__
real wrf_smag_w_ty(const WrfSmagGrid& q, const real* km, const real* dy23,
                   int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    real rhoavg = 0.5f * (q.fnm[kw] * (wrf_rho(q, kw, j - 1, i)
                                        + wrf_rho(q, kw, j, i))
                           + q.fnp[kw] * (wrf_rho(q, kw - 1, j - 1, i)
                                        + wrf_rho(q, kw - 1, j, i)));
    real kavg = 0.5f * (q.fnm[kw] * (wrf_k(q, km, kw, j - 1, i)
                                      + wrf_k(q, km, kw, j, i))
                         + q.fnp[kw] * (wrf_k(q, km, kw - 1, j - 1, i)
                                      + wrf_k(q, km, kw - 1, j, i)));
    return -rhoavg * kavg * dy23[((SmagIndex)kw * (q.ny + 1) + j) * q.nx + i];
}

// smag2d.cu's wrf_tau13_mavg / wrf_tau23_mavg on the stored faces, in each
// variant's own summation order.
__device__ __forceinline__
real wrf_smag_w_tx_mavg(const WrfSmagGrid& q, const real* km, const real* dx13, int k, int j, int i)
#ifdef GPUWM_WRF_EXACT_C_DIFFUSION
{return 0.25f*(wrf_smag_w_tx(q,km,dx13,k+1,j,i+1)+wrf_smag_w_tx(q,km,dx13,k+1,j,i)+wrf_smag_w_tx(q,km,dx13,k,j,i+1)+wrf_smag_w_tx(q,km,dx13,k,j,i));}
#else
{
    return 0.25f * (wrf_smag_w_tx(q,km,dx13,k,j,i)
                     + wrf_smag_w_tx(q,km,dx13,k,j,i + 1)
                     + wrf_smag_w_tx(q,km,dx13,k + 1,j,i)
                     + wrf_smag_w_tx(q,km,dx13,k + 1,j,i + 1));
}
#endif

__device__ __forceinline__
real wrf_smag_w_ty_mavg(const WrfSmagGrid& q, const real* km, const real* dy23, int k, int j, int i)
#ifdef GPUWM_WRF_EXACT_C_DIFFUSION
{return 0.25f*(wrf_smag_w_ty(q,km,dy23,k+1,j+1,i)+wrf_smag_w_ty(q,km,dy23,k+1,j,i)+wrf_smag_w_ty(q,km,dy23,k,j+1,i)+wrf_smag_w_ty(q,km,dy23,k,j,i));}
#else
{
    return 0.25f * (wrf_smag_w_ty(q,km,dy23,k,j,i)
                     + wrf_smag_w_ty(q,km,dy23,k,j + 1,i)
                     + wrf_smag_w_ty(q,km,dy23,k + 1,j,i)
                     + wrf_smag_w_ty(q,km,dy23,k + 1,j + 1,i));
}
#endif

extern "C" __global__
void wrf_smag_hd_w_faces(WRF_SMAG_GRID_ARGS, const real* km,
                         const real* dx13, const real* dy23,
                         real* tend,
                         int nz, int ny, int nx, int phb3d,
                         int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int kw = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || kw >= nz + 1 || kw == 0 || kw == nz) return;
    WRF_SMAG_MAKE_GRID;
    int ii = wrf_ix(q, i), jj = wrf_iy(q, j);
    real map = msft[(SmagIndex)jj * nx + ii];
    real zx_w = 0.5f * (wrf_zx(q, kw, j, i) + wrf_zx(q, kw, j, i + 1));
    real zy_w = 0.5f * (wrf_zy(q, kw, j, i) + wrf_zy(q, kw, j + 1, i));
    real rz = wrf_rdz(q, kw, j, i);
    real divh = map * rdx * (wrf_smag_w_tx(q,km,dx13,kw,j,i + 1)
                              - wrf_smag_w_tx(q,km,dx13,kw,j,i))
              + map * rdy * (wrf_smag_w_ty(q,km,dy23,kw,j + 1,i)
                              - wrf_smag_w_ty(q,km,dy23,kw,j,i));
    real divz = map * rz * (zx_w * (wrf_smag_w_tx_mavg(q, km, dx13, kw, j, i)
                                     - wrf_smag_w_tx_mavg(q, km, dx13, kw - 1, j, i))
                           + zy_w * (wrf_smag_w_ty_mavg(q, km, dy23, kw, j, i)
                                     - wrf_smag_w_ty_mavg(q, km, dy23, kw - 1, j, i)));
    tend[I3(kw, j, i, ny, nx)] += G / (dn[kw] * rz) * (divh - divz);
}

#undef WRF_SMAG_MAKE_GRID
#undef WRF_SMAG_GRID_ARGS

#endif
