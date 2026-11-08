// WOOF-OPTIN: nemo
// Derived from NEMO 5.0.2 (CeCILL v2.0); see optional/nemo/LICENSE and
// NOTICE.md.  Not part of the default build or of any wheel.
//
// NEMO's iso-level lateral diffusion of a tracer, src/OCE/TRA/traldf_lev.F90:
// traldf_lev_lap (laplacian, added with a plus sign) and traldf_lev_blp
// (bilaplacian: the laplacian of the laplacian, subtracted), given the
// eddy diffusivities ahtu/ahtv.  Double, NEMO's expressions in NEMO's order;
// every intermediate (zaheeu, zfu, zlap) is the same word NEMO stores, here
// recomputed per use.  Fields are (jpk, jpj, jpi) C order with NEMO 5.0's
// two-cell halo; one thread per interior column, 1-based indices.

#define I3(i, j, k) ((size_t)((k) - 1) * n2 + (size_t)((j) - 1) * nx + ((i) - 1))
#define I2(i, j) ((size_t)((j) - 1) * nx + ((i) - 1))

struct Ldf {
    const double *ahtu, *ahtv, *e2_e1u, *e1_e2v, *e3u, *e3v, *e3t, *r1_e1e2t, *tb;
    int nx; size_t n2;
};

__device__ static double ldf_zfu(const Ldf& g, int i, int j, int k)
{
    int nx = g.nx; size_t n2 = g.n2;
    double zaheeu = g.ahtu[I3(i, j, k)] * g.e2_e1u[I2(i, j)] * g.e3u[I3(i, j, k)];
    return zaheeu * ( g.tb[I3(i + 1, j, k)] - g.tb[I3(i, j, k)] );
}
__device__ static double ldf_zfv(const Ldf& g, int i, int j, int k)
{
    int nx = g.nx; size_t n2 = g.n2;
    double zaheev = g.ahtv[I3(i, j, k)] * g.e1_e2v[I2(i, j)] * g.e3v[I3(i, j, k)];
    return zaheev * ( g.tb[I3(i, j + 1, k)] - g.tb[I3(i, j, k)] );
}
__device__ static double ldf_zlap(const Ldf& g, int i, int j, int k)
{
    int nx = g.nx; size_t n2 = g.n2;
    return (  ( ldf_zfu(g, i, j, k) - ldf_zfu(g, i - 1, j, k) )
            + ( ldf_zfv(g, i, j, k) - ldf_zfv(g, i, j - 1, k) )  )
         *  g.r1_e1e2t[I2(i, j)] /  g.e3t[I3(i, j, k)];
}
__device__ static double ldf_zfu2(const Ldf& g, int i, int j, int k)
{
    int nx = g.nx; size_t n2 = g.n2;
    double zaheeu = g.ahtu[I3(i, j, k)] * g.e2_e1u[I2(i, j)] * g.e3u[I3(i, j, k)];
    return zaheeu * ( ldf_zlap(g, i + 1, j, k) - ldf_zlap(g, i, j, k) );
}
__device__ static double ldf_zfv2(const Ldf& g, int i, int j, int k)
{
    int nx = g.nx; size_t n2 = g.n2;
    double zaheev = g.ahtv[I3(i, j, k)] * g.e1_e2v[I2(i, j)] * g.e3v[I3(i, j, k)];
    return zaheev * ( ldf_zlap(g, i, j + 1, k) - ldf_zlap(g, i, j, k) );
}

extern "C" __global__
void nemo_traldf_lev(const double* __restrict__ ahtu, const double* __restrict__ ahtv,
                     const double* __restrict__ e2_e1u, const double* __restrict__ e1_e2v,
                     const double* __restrict__ e3u, const double* __restrict__ e3v,
                     const double* __restrict__ e3t, const double* __restrict__ r1_e1e2t,
                     const double* __restrict__ tb, double* __restrict__ rhs, int bilaplacian,
                     int nx, int ny, int jpk, int H)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int ni = nx - 2 * H, nj = ny - 2 * H;
    if (t >= ni * nj) return;
    int i = t % ni + H + 1, j = t / ni + H + 1;
    size_t n2 = (size_t)nx * ny;
    Ldf g{ahtu, ahtv, e2_e1u, e1_e2v, e3u, e3v, e3t, r1_e1e2t, tb, nx, n2};
    for (int k = 1; k <= jpk - 1; ++k) {
        size_t o = I3(i, j, k);
        if (!bilaplacian)
            rhs[o] = rhs[o]
                   + (  ( ldf_zfu(g, i, j, k) - ldf_zfu(g, i - 1, j, k) )
                      + ( ldf_zfv(g, i, j, k) - ldf_zfv(g, i, j - 1, k) )  )
                   * r1_e1e2t[I2(i, j)] / e3t[o];
        else
            rhs[o] = rhs[o]
                   - (  ( ldf_zfu2(g, i, j, k) - ldf_zfu2(g, i - 1, j, k) )
                      + ( ldf_zfv2(g, i, j, k) - ldf_zfv2(g, i, j - 1, k) )  )
                   *  r1_e1e2t[I2(i, j)] /  e3t[o];
    }
}
