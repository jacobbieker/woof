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
// every face stress recomputes its deformation from the winds, so one face
// was evaluated up to twelve times: 13.1 ms of a 76 ms step at
// 200x200x50 on an RTX A6000.  wrf_smag_w_tau_faces stores each face once;
// wrf_smag_hd_w_faces is wrf_smag_hd_w's statement with the stored words in
// place of the calls, in the same order.  Same options as smag2d
// (DIFFUSION_OPTIONS: --fmad=false --ftz=false --prec-div=true
// --prec-sqrt=true): no contraction across what used to be the inlined call
// boundary, so a stored word is the word the call returned.
//
// Defined where wrf_smag_hd_w takes its plain (#else) statement:
// GPUWM_WRF_EXACT or GPUWM_WRF_EXACT_C_DIFFUSION.  Buffers: tx (nz, ny,
// nx + 1) for i = 0..nx, ty (nz, ny + 1, nx) for j = 0..ny, level k = 0..nz-1;
// level nz is wrf_tau13/wrf_tau23's own 0.0f (kw >= nz) and is not stored.

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
void wrf_smag_w_tau_faces(WRF_SMAG_GRID_ARGS, const real* km,
                          real* tx, real* ty,
                          int nz, int ny, int nx, int phb3d,
                          int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i > nx || j > ny || k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    if (j < ny)
        tx[((SmagIndex)k * ny + j) * (nx + 1) + i] = wrf_tau13(q, km, k, j, i);
    if (i < nx)
        ty[((SmagIndex)k * (ny + 1) + j) * nx + i] = wrf_tau23(q, km, k, j, i);
}

__device__ __forceinline__
real wrf_smag_w_tx(const real* tx, int k, int j, int i, int nz, int ny, int nx)
{
    return k >= nz ? 0.0f : tx[((SmagIndex)k * ny + j) * (nx + 1) + i];
}

__device__ __forceinline__
real wrf_smag_w_ty(const real* ty, int k, int j, int i, int nz, int ny, int nx)
{
    return k >= nz ? 0.0f : ty[((SmagIndex)k * (ny + 1) + j) * nx + i];
}

// smag2d.cu's wrf_tau13_mavg / wrf_tau23_mavg on the stored faces, in each
// variant's own summation order.
__device__ __forceinline__
real wrf_smag_w_tx_mavg(const real* tx, int k, int j, int i, int nz, int ny, int nx)
#ifdef GPUWM_WRF_EXACT_C_DIFFUSION
{return 0.25f*(wrf_smag_w_tx(tx,k+1,j,i+1,nz,ny,nx)+wrf_smag_w_tx(tx,k+1,j,i,nz,ny,nx)+wrf_smag_w_tx(tx,k,j,i+1,nz,ny,nx)+wrf_smag_w_tx(tx,k,j,i,nz,ny,nx));}
#else
{
    return 0.25f * (wrf_smag_w_tx(tx, k, j, i, nz, ny, nx)
                     + wrf_smag_w_tx(tx, k, j, i + 1, nz, ny, nx)
                     + wrf_smag_w_tx(tx, k + 1, j, i, nz, ny, nx)
                     + wrf_smag_w_tx(tx, k + 1, j, i + 1, nz, ny, nx));
}
#endif

__device__ __forceinline__
real wrf_smag_w_ty_mavg(const real* ty, int k, int j, int i, int nz, int ny, int nx)
#ifdef GPUWM_WRF_EXACT_C_DIFFUSION
{return 0.25f*(wrf_smag_w_ty(ty,k+1,j+1,i,nz,ny,nx)+wrf_smag_w_ty(ty,k+1,j,i,nz,ny,nx)+wrf_smag_w_ty(ty,k,j+1,i,nz,ny,nx)+wrf_smag_w_ty(ty,k,j,i,nz,ny,nx));}
#else
{
    return 0.25f * (wrf_smag_w_ty(ty, k, j, i, nz, ny, nx)
                     + wrf_smag_w_ty(ty, k, j + 1, i, nz, ny, nx)
                     + wrf_smag_w_ty(ty, k + 1, j, i, nz, ny, nx)
                     + wrf_smag_w_ty(ty, k + 1, j + 1, i, nz, ny, nx));
}
#endif

extern "C" __global__
void wrf_smag_hd_w_faces(WRF_SMAG_GRID_ARGS, const real* tx, const real* ty,
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
    real divh = map * rdx * (wrf_smag_w_tx(tx, kw, j, i + 1, nz, ny, nx)
                              - wrf_smag_w_tx(tx, kw, j, i, nz, ny, nx))
              + map * rdy * (wrf_smag_w_ty(ty, kw, j + 1, i, nz, ny, nx)
                              - wrf_smag_w_ty(ty, kw, j, i, nz, ny, nx));
    real divz = map * rz * (zx_w * (wrf_smag_w_tx_mavg(tx, kw, j, i, nz, ny, nx)
                                     - wrf_smag_w_tx_mavg(tx, kw - 1, j, i, nz, ny, nx))
                           + zy_w * (wrf_smag_w_ty_mavg(ty, kw, j, i, nz, ny, nx)
                                     - wrf_smag_w_ty_mavg(ty, kw - 1, j, i, nz, ny, nx)));
    tend[I3(kw, j, i, ny, nx)] += G / (dn[kw] * rz) * (divh - divz);
}

#undef WRF_SMAG_MAKE_GRID
#undef WRF_SMAG_GRID_ARGS

#endif
