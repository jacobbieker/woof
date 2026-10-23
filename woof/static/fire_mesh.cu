// The fire mesh of a WRF-SFIRE-style fire model, from WRF v4.7.1's fire code.
//
//   fire_zsf      phys/module_fr_fire_util.F interpolate_2d, as called by
//                 interpolate_z2fire: the atmospheric terrain, continued one
//                 cell past the domain (continue_at_boundary, bias 0), is
//                 refined bilinearly onto the fire mesh nodes,
//                     v1 = (1-tx)*(1-ty)*v2(i2,j2) + (1-tx)*ty*v2(i2,j2+1)
//                        + tx*(1-ty)*v2(i2+1,j2) + tx*ty*v2(i2+1,j2+1)
//                 left to right.  The host gives each fire column its
//                 coarse cell i2 and weight tx, and each fire row its j2
//                 and ty (woof/static/fuel.py: wrf_refinement_axis).  The
//                 cell is the last one WRF's loop writes the node from,
//                 which matters for odd ratios, where a node sits on a cell
//                 edge.
//   fire_slope    phys/module_fr_fire_model.F:194-208: ZSF continued one
//                 node past the mesh (continue_at_boundary, bias 0), then
//                 (zsf(i+1,j)-zsf(i-1,j))/(2.*fdx), and the same in y.
//   fire_repeat   each fire cell takes its parent atmospheric cell's
//                 category.
//   fire_gather   a LANDFIRE pixel per fire cell, by the pixel indices the
//                 host computed, or the caller's missing code where the
//                 host marked the cell outside the raster (row < 0).
//
// REAL(4) throughout, in WRF's order of evaluation.  The module is compiled
// with -fmad=false --ftz=false (woof/static/fire_mesh_gpu.py), so no product
// is fused into a sum and subnormals stay, as gfortran at -O0 does.
//
// A static-preparation unit, not a forecast kernel: it runs once, before a
// forecast's device state exists, so it lives beside its loader rather than
// in woof/core/kernels, whose sources the forecast's VRAM frame pricing and
// freeze pins enumerate.  It needs no preamble.
// Layout: row-major (j, i); the halo-extended atmospheric terrain is
// (ny + 2, nx + 2) with the domain at [1..ny][1..nx].

__device__ __forceinline__ float fm_ex(float a, float b)
{
    // continue_at_boundary's EX with bias = 0:
    // (1.-bias)*(2.*a-b)+bias*max(2.*a-b,a,b)
    const float bias = 0.0f;
    const float t = 2.0f * a - b;
    return (1.0f - bias) * t + bias * fmaxf(fmaxf(t, a), b);
}

extern "C" __global__
void fire_zsf(const float* __restrict__ za, int nxe,
              const int* __restrict__ i2, const float* __restrict__ tx,
              const int* __restrict__ j2, const float* __restrict__ ty,
              int nxf, int nyf, float* __restrict__ zsf)
{
    const long long n = (long long)nxf * nyf;
    for (long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         k < n; k += (long long)gridDim.x * blockDim.x) {
        const int i = (int)(k % nxf);
        const int j = (int)(k / nxf);
        const int ci = i2[i];
        const int cj = j2[j];
        const float x = tx[i];
        const float y = ty[j];
        const float v00 = za[(long long)cj * nxe + ci];
        const float v01 = za[(long long)(cj + 1) * nxe + ci];
        const float v10 = za[(long long)cj * nxe + ci + 1];
        const float v11 = za[(long long)(cj + 1) * nxe + ci + 1];
        float v = (1.0f - x) * (1.0f - y) * v00;
        v = v + (1.0f - x) * y * v01;
        v = v + x * (1.0f - y) * v10;
        v = v + x * y * v11;
        zsf[k] = v;
    }
}

extern "C" __global__
void fire_slope(const float* __restrict__ zsf, int nxf, int nyf,
                float twodx, float twody,
                float* __restrict__ dzdx, float* __restrict__ dzdy)
{
    const long long n = (long long)nxf * nyf;
    for (long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         k < n; k += (long long)gridDim.x * blockDim.x) {
        const int i = (int)(k % nxf);
        const int j = (int)(k / nxf);
        const long long row = (long long)j * nxf;
        float west, east, south, north;
        // nxf, nyf >= 2: fire_mesh_gpu.slope refuses a one-node axis.
        west = (i == 0) ? fm_ex(zsf[row], zsf[row + 1]) : zsf[row + i - 1];
        east = (i == nxf - 1) ? fm_ex(zsf[row + nxf - 1], zsf[row + nxf - 2])
                              : zsf[row + i + 1];
        south = (j == 0) ? fm_ex(zsf[i], zsf[(long long)nxf + i])
                         : zsf[row - nxf + i];
        north = (j == nyf - 1)
                    ? fm_ex(zsf[row + i], zsf[row - nxf + i])
                    : zsf[row + nxf + i];
        dzdx[k] = (east - west) / twodx;
        dzdy[k] = (north - south) / twody;
    }
}

extern "C" __global__
void fire_repeat(const unsigned char* __restrict__ atm, int nx,
                 int sr_x, int sr_y, int nxf, int nyf,
                 unsigned char* __restrict__ fire)
{
    const long long n = (long long)nxf * nyf;
    for (long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         k < n; k += (long long)gridDim.x * blockDim.x) {
        const int i = (int)(k % nxf);
        const int j = (int)(k / nxf);
        fire[k] = atm[(long long)(j / sr_y) * nx + i / sr_x];
    }
}

extern "C" __global__
void fire_gather(const int* __restrict__ block, int width,
                 const int* __restrict__ row, const int* __restrict__ col,
                 long long n, int missing, int* __restrict__ codes)
{
    for (long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         k < n; k += (long long)gridDim.x * blockDim.x) {
        const int r = row[k];
        codes[k] = (r < 0) ? missing
                           : block[(long long)r * width + col[k]];
    }
}
