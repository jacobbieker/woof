// Porous immersed obstacles (woof's opt-in obstacle_opt = 1; not WRF).
// One launch per RK stage over the sparse list of momentum faces that touch
// an obstacle: u faces first (nu), then v (nv), then w (nw).  Each face
// carries a penalisation weight a (dimensionless; a/dt is the rate) and a
// drag coefficient b (1/m, canopy form drag plus wall stress), so the rate
// is k = a/dt + b|U| and the coupled tendency takes
//     tend -= k / (1 + k dt) * (coupled momentum)
// (the implicit-rate form: bounded by |momentum|/dt for any k).  |U| uses
// the stage winds averaged onto the face.  Scalar operation order, round to
// nearest on every operation, no contraction.

__device__ __forceinline__ int obs_wrap(int i, int n, int periodic) {
    if (periodic) return (i % n + n) % n;
    return min(max(i, 0), n - 1);
}

__device__ __forceinline__ float obs_avg4(float a, float b, float c, float d) {
    return __fmul_rn(0.25f, __fadd_rn(__fadd_rn(a, b), __fadd_rn(c, d)));
}

extern "C" __global__ void obstacle_drag(
    int nx, int ny, int nz, int nu, int nv, int nw,
    int periodic_x, int periodic_y, float dt,
    const int *face, const float *a, const float *b,
    const float *u, const float *v, const float *w,
    const float *ru, const float *rv,
    const float *mut, const float *c1f, const float *c2f,
    const float *msft, int w_divides_msft,
    float *ru_t, float *rv_t, float *rw_t) {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= nu + nv + nw) return;
    int idx = face[n];
    int nxu = nx + 1, nyv = ny + 1;
    float uu, vv, ww, momentum;
    float *tend;
    if (n < nu) {                       // u(k, j, i), i in 0..nx
        int i = idx % nxu, j = (idx / nxu) % ny, k = idx / (nxu * ny);
        int l = obs_wrap(i - 1, nx, periodic_x), r = obs_wrap(i, nx, periodic_x);
        uu = u[idx];
        vv = obs_avg4(v[(k * nyv + j) * nx + l], v[(k * nyv + j) * nx + r],
                      v[(k * nyv + j + 1) * nx + l], v[(k * nyv + j + 1) * nx + r]);
        ww = obs_avg4(w[(k * ny + j) * nx + l], w[(k * ny + j) * nx + r],
                      w[((k + 1) * ny + j) * nx + l], w[((k + 1) * ny + j) * nx + r]);
        momentum = ru[idx];
        tend = ru_t;
    } else if (n < nu + nv) {           // v(k, j, i), j in 0..ny
        int i = idx % nx, j = (idx / nx) % nyv, k = idx / (nx * nyv);
        int s = obs_wrap(j - 1, ny, periodic_y), t = obs_wrap(j, ny, periodic_y);
        vv = v[idx];
        uu = obs_avg4(u[(k * ny + s) * nxu + i], u[(k * ny + s) * nxu + i + 1],
                      u[(k * ny + t) * nxu + i], u[(k * ny + t) * nxu + i + 1]);
        ww = obs_avg4(w[(k * ny + s) * nx + i], w[(k * ny + t) * nx + i],
                      w[((k + 1) * ny + s) * nx + i], w[((k + 1) * ny + t) * nx + i]);
        momentum = rv[idx];
        tend = rv_t;
    } else {                            // w(k, j, i), k in 1..nz-1
        int i = idx % nx, j = (idx / nx) % ny, k = idx / (nx * ny);
        int col = j * nx + i;
        ww = w[idx];
        uu = obs_avg4(u[((k - 1) * ny + j) * nxu + i], u[((k - 1) * ny + j) * nxu + i + 1],
                      u[(k * ny + j) * nxu + i], u[(k * ny + j) * nxu + i + 1]);
        vv = obs_avg4(v[((k - 1) * nyv + j) * nx + i], v[((k - 1) * nyv + j + 1) * nx + i],
                      v[(k * nyv + j) * nx + i], v[(k * nyv + j + 1) * nx + i]);
        momentum = __fmul_rn(__fadd_rn(__fmul_rn(c1f[k], mut[col]), c2f[k]), ww);
        if (w_divides_msft) momentum = __fdiv_rn(momentum, msft[col]);
        tend = rw_t;
    }
    float speed = __fsqrt_rn(__fadd_rn(__fadd_rn(__fmul_rn(uu, uu), __fmul_rn(vv, vv)),
                                       __fmul_rn(ww, ww)));
    float rate = __fadd_rn(__fdiv_rn(a[n], dt), __fmul_rn(b[n], speed));
    if (!(rate > 0.0f)) return;
    float keff = __fdiv_rn(rate, __fadd_rn(1.0f, __fmul_rn(rate, dt)));
    tend[idx] = __fsub_rn(tend[idx], __fmul_rn(keff, momentum));
}
