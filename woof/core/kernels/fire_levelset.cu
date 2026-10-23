// Level-set fire spread, fuel burn and heat release from WRF v4.7.1's fire
// model (the WRF-SFIRE lineage), REAL(4) word for word.
//
//   fl_extend          phys/module_fr_fire_util.F continue_at_boundary: the
//                      one-cell ring beyond the fire domain, extrapolated and
//                      max-guarded (EX, bias = fire_lfn_ext_up).
//   fl_tend_stage      phys/module_fr_fire_core.F tend_ls (fire_upwinding = 9,
//                      ENO1 in the ten-node boundary band and outside the
//                      +-fire_lsm_band_ngp*dx band, WENO5 inside it) with
//                      fire_ros (phys/module_fr_fire_phys.F) inlined, and one
//                      Runge-Kutta stage of prop_ls_rk3: out = lfn_0 + c*tend.
//   fl_bound_reduce    tend_ls's CFL bound, 1/(max + tol), from
//                      fl_tend_stage's per-block maxima, and prop_ls_rk3's
//                      min over its three stages.
//   fl_tign_update     tign_update (its boundary_guard stop is not a field).
//   fl_reinit_init /
//   fl_reinit_stage /
//   fl_reinit_final    reinit_ls_rk3 and advance_ls_reinit
//                      (fire_upwinding_reinit = 4), then fire_model pass 5's
//                      copy lfn = lfn_out.
//   fl_ignite          ignite_fire with nearest inlined, one ignition line.
//   fl_ignite_xy       the same at given node coordinates (fxlong/fxlat).
//   fl_fuel_heat       fuel_left (fuel_left_method = 1, irl = jrl = 2,
//                      fuel_left_cell_1 inlined), fire_model pass 6's
//                      fuel_frac_burnt, and heat_fluxes.
//   fl_sum_cells       sum_2d_cells (module_fr_fire_util.F) and the driver's
//                      scaling (module_fr_fire_driver.F:941-953).
//   fl_accumulate /
//   fl_scale           the substep mean of an atmospheric flux.
//
// Every statement follows WRF's own in WRF's order of evaluation.  The unit
// is compiled with -fmad=false --ftz=false by woof/fire/levelset.py, so no
// product is fused into a sum (gfortran -O0 fuses none) and subnormals stay.
// Divisions and square roots are spelled __fdiv_rn/__fsqrt_rn.  Fortran's
// real powers (x**2.0, x**(-0.3), x**bbb) are glibc's powf (gfk_pow):
// gfortran calls powf for a real exponent even when it is 2.0; integer powers
// are products.  EXP is glibc's expf (gfk_exp).  MAX/MIN are fmaxf/fminf.
// The constants WRF folds at compile time (13.0/12.0, 1./10., 100*eps, ...)
// are written as the float32 words gfortran folds them to.
//
// Layout: the level-set function, ignition time and the reinitialization
// stages are (ny + 2, nx + 2) row-major arrays, WRF's (i, j) memory with one
// halo node each side: node (i, j), 1 <= i <= nx, 1 <= j <= ny, at
// [j * (nx + 2) + i].  The halo of the state lfn and tign is never written:
// it holds WRF's zero fill, which fuel_left reads at the domain edge.
// Every other field is (ny, nx) at [(j - 1) * nx + (i - 1)].  One thread
// owns one node.

#define FL_BDY 10

__device__ __forceinline__ float fl_div(float a, float b)
{ return __fdiv_rn(a, b); }

__device__ __forceinline__ float fl_sqrt(float a)
{ return __fsqrt_rn(a); }

// continue_at_boundary's statement function EX.
__device__ __forceinline__ float fl_ex(float a, float b, float bias)
{
    float lin = 2.0f * a - b;
    return (1.0f - bias) * lin + bias * fmaxf(fmaxf(lin, a), b);
}

__device__ float fl_select_eno(float dl, float dr)
{
    float d;
    if (!(dl > 0.0f) && !(dr > 0.0f)) {
        d = dr;
    } else if (!(dl < 0.0f) && !(dr < 0.0f)) {
        d = dl;
    } else if (!(dl < 0.0f) && !(dr > 0.0f)) {
        if (!(fabsf(dr) < fabsf(dl))) d = dr; else d = dl;
    } else {
        d = 0.0f;
    }
    return d;
}

__device__ float fl_select_4th(float dx, float li, float lim1, float lim2,
                               float lip1, float lip2)
{
    float p = fl_div(7.0f * lip1 + 7.0f * li - lip2 - lim1, 12.0f * dx);
    float m = fl_div(7.0f * li + 7.0f * lim1 - lip1 - lim2, 12.0f * dx);
    return p - m;
}

__device__ __forceinline__ float fl_sq(float x) { return x * x; }

__device__ float fl_select_weno5(float dx, float it, float im1t, float im2t,
                                 float im3t, float ip1t, float ip2t,
                                 float ip3t, float uf)
{
    const float g1 = 0x1.99999ap-4f;      // 1./10.
    const float g2 = 0x1.333334p-1f;      // 3./5.
    const float g3 = 0x1.333334p-2f;      // 3./10.
    const float c1312 = 0x1.155556p+0f;   // 13.0/12.0
    const float tol = 1e-6f;
    float li, lim1, lim2, lim3, lip1, lip2;
    if (uf >= 0.0f) {
        li = it; lim1 = im1t; lim2 = im2t; lim3 = im3t; lip1 = ip1t;
        lip2 = ip2t;
    } else {
        li = it; lim1 = ip1t; lim2 = ip2t; lim3 = ip3t; lip1 = im1t;
        lip2 = im2t;
    }
    // numerical flux at the i+1/2 face
    float fh1 = fl_div(2.0f * lim2 - 7.0f * lim1 + 11.0f * li, 6.0f);
    float fh2 = fl_div(-1.0f * lim1 + 5.0f * li + 2.0f * lip1, 6.0f);
    float fh3 = fl_div(2.0f * li + 5.0f * lip1 - 1.0f * lip2, 6.0f);
    float b1 = c1312 * fl_sq(lim2 - 2.0f * lim1 + li)
             + 0.25f * fl_sq(lim2 - 4.0f * lim1 + 3.0f * li);
    float b2 = c1312 * fl_sq(lim1 - 2.0f * li + lip1)
             + 0.25f * fl_sq(lim1 - lip1);
    float b3 = c1312 * fl_sq(li - 2.0f * lip1 + lip2)
             + 0.25f * fl_sq(3.0f * li - 4.0f * lip1 + lip2);
    float w1t = fl_div(g1, fl_sq(b1 + tol));
    float w2t = fl_div(g2, fl_sq(b2 + tol));
    float w3t = fl_div(g3, fl_sq(b3 + tol));
    float w1 = fl_div(w1t, w1t + w2t + w3t);
    float w2 = fl_div(w2t, w1t + w2t + w3t);
    float w3 = fl_div(w3t, w1t + w2t + w3t);
    float flux_p = w1 * fh1 + w2 * fh2 + w3 * fh3;
    // numerical flux at the i-1/2 face
    fh1 = fl_div(2.0f * lim3 - 7.0f * lim2 + 11.0f * lim1, 6.0f);
    fh2 = fl_div(-1.0f * lim2 + 5.0f * lim1 + 2.0f * li, 6.0f);
    fh3 = fl_div(2.0f * lim1 + 5.0f * li - 1.0f * lip1, 6.0f);
    b1 = c1312 * fl_sq(lim3 - 2.0f * lim2 + lim1)
       + 0.25f * fl_sq(lim3 - 4.0f * lim2 + 3.0f * lim1);
    b2 = c1312 * fl_sq(lim2 - 2.0f * lim1 + li) + 0.25f * fl_sq(lim2 - li);
    b3 = c1312 * fl_sq(lim1 - 2.0f * li + lip1)
       + 0.25f * fl_sq(3.0f * lim1 - 4.0f * li + lip1);
    w1t = fl_div(g1, fl_sq(b1 + tol));
    w2t = fl_div(g2, fl_sq(b2 + tol));
    w3t = fl_div(g3, fl_sq(b3 + tol));
    w1 = fl_div(w1t, w1t + w2t + w3t);
    w2 = fl_div(w2t, w1t + w2t + w3t);
    w3 = fl_div(w3t, w1t + w2t + w3t);
    float flux_m = w1 * fh1 + w2 * fh2 + w3 * fh3;
    if (uf >= 0.0f) return fl_div(flux_p - flux_m, dx);
    return fl_div(flux_m - flux_p, dx);
}

// fire_ros, ibeh = 1 (Rothermel), not chaparral.  WRF's chaparral arm
// leaves ros_base undefined, so it is not ported (woof/fire/levelset.py).
__device__ void fl_fire_ros(float *ros_base, float *ros_wind,
                            float *ros_slope, float nvx, float nvy,
                            float vx, float vy, float dzdx, float dzdy,
                            float r0, float bbb, float phiwc, float betafl,
                            float ischap, int advection)
{
    const float tiny = 0x1p-126f;
    float speed, tanphi, cor_wind, cor_slope;
    if (advection != 0) {
        speed = fl_sqrt(vx * vx + vy * vy) + tiny;
        tanphi = fl_sqrt(dzdx * dzdx + dzdy * dzdy) + tiny;
        cor_wind = fmaxf(0.0f, fl_div(vx * nvx + vy * nvy, speed));
        cor_slope = fmaxf(0.0f, fl_div(dzdx * nvx + dzdy * nvy, tanphi));
    } else {
        speed = vx * nvx + vy * nvy;
        tanphi = dzdx * nvx + dzdy * nvy;
        cor_wind = 1.0f;
        cor_slope = 1.0f;
    }
    float spdms = fmaxf(speed, 0.0f);
    float umidm = fminf(spdms, 30.0f);
    float umid = umidm * 196.850f;
    float phiw = gfk_pow(umid, bbb) * phiwc;
    float phis = 0.0f;
    if (tanphi > 0.0f) {
        phis = 5.275f * gfk_pow(betafl, -0.3f) * (tanphi * tanphi);
    }
    float rb = r0 * 0.00508f;
    float rw = rb * phiw;
    float rs = rb * phis;
    if (ischap > 0.0f) {
        // fire_ros's chaparral arm: the wind alone, with a backing floor.
        // WRF never assigns ros_base on it (the caller's variable is read
        // undefined); the port fixes it at 0, as woof.fire.wrf_sfire does.
        rb = 0.0f;
        rw = fmaxf(1.2974f * gfk_pow(spdms, 1.41f), 0.03333f);
        rs = 0.0f;
    }
    rw = rw * cor_wind;
    rs = rs * cor_slope;
    float excess = rb + rw + rs - 6.0f;
    if (excess > 0.0f) {
        rw = rw - fl_div(excess * rw, rw + rs);
        rs = rs - fl_div(excess * rs, rw + rs);
    }
    *ros_base = rb;
    *ros_wind = rw;
    *ros_slope = rs;
}

extern "C" __global__ void fl_extend(float *a, float bias, int nx, int ny)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int p = nx + 2;
    if (k < ny) {                                   // x edges, j = 1..ny
        int j = k + 1;
        a[j * p] = fl_ex(a[j * p + 1], a[j * p + 2], bias);
        a[j * p + nx + 1] = fl_ex(a[j * p + nx], a[j * p + nx - 1], bias);
    } else if (k < ny + nx) {                       // y edges, i = 1..nx
        int i = k - ny + 1;
        a[i] = fl_ex(a[p + i], a[2 * p + i], bias);
        a[(ny + 1) * p + i] = fl_ex(a[ny * p + i], a[(ny - 1) * p + i], bias);
    } else if (k == ny + nx) {                      // the four corners
        a[0] = fl_ex(a[p + 1], a[2 * p + 2], bias);
        a[(ny + 1) * p] = fl_ex(a[ny * p + 1], a[(ny - 1) * p + 2], bias);
        a[nx + 1] = fl_ex(a[p + nx], a[2 * p + nx - 1], bias);
        a[(ny + 1) * p + nx + 1] =
            fl_ex(a[ny * p + nx], a[(ny - 1) * p + nx - 1], bias);
    }
}

// The node's tend_ls, then out = lfn_0 + coef * tend.  fl_tend_stage reduces
// the CFL bound's running MAX per block into partial[block] (MAX is exact in
// any order) and fl_bound_reduce finishes it.  FL_TPB threads a block.
#define FL_TPB 256

__device__ void fl_tend_node(
    const float *__restrict__ lfn, const float *__restrict__ lfn0,
    float *__restrict__ out, float *__restrict__ ros, float *cb_out,
    const float *__restrict__ vx,
    const float *__restrict__ vy, const float *__restrict__ dzdx,
    const float *__restrict__ dzdy, const float *__restrict__ r0,
    const float *__restrict__ bbb, const float *__restrict__ phiwc,
    const float *__restrict__ betafl, const float *__restrict__ ischap,
    float dx, float dy, float coef,
    float thr_hl, float thr_av, float visc, float visc_bg, float visc_band,
    float slope_factor, int advection, int nx, int ny, int i, int j)
{
    const float eps = 0x1p-23f;                     // epsilon(0.0)
    int p = nx + 2;
    int c = j * p + i;
    int f = (j - 1) * nx + (i - 1);
    float l = lfn[c];
    float drx = fl_div(lfn[c + 1] - l, dx);
    float dlx = fl_div(l - lfn[c - 1], dx);
    float dry = fl_div(lfn[c + p] - l, dy);
    float dly = fl_div(l - lfn[c - p], dy);
    float d2x, d2y;
    bool band = i < 1 + FL_BDY || i > nx - FL_BDY || j < 1 + FL_BDY
             || j > ny - FL_BDY;
    if (band || !(fabsf(l) < thr_hl)) {
        d2x = fl_select_eno(dlx, drx);
        d2y = fl_select_eno(dly, dry);
    } else {
        float s4x = fl_select_4th(dx, l, lfn[c - 1], lfn[c - 2], lfn[c + 1],
                                  lfn[c + 2]);
        float s4y = fl_select_4th(dy, l, lfn[c - p], lfn[c - 2 * p],
                                  lfn[c + p], lfn[c + 2 * p]);
        float a = s4x * vx[f] + s4y * vy[f];
        float sx = a * s4x;
        float sy = a * s4y;
        d2x = fl_select_weno5(dx, l, lfn[c - 1], lfn[c - 2], lfn[c - 3],
                              lfn[c + 1], lfn[c + 2], lfn[c + 3], sx);
        d2y = fl_select_weno5(dy, l, lfn[c - p], lfn[c - 2 * p],
                              lfn[c - 3 * p], lfn[c + p], lfn[c + 2 * p],
                              lfn[c + 3 * p], sy);
    }
    float grad = fl_sqrt(d2x * d2x + d2y * d2y);
    float scale = fl_sqrt(gfk_pow(grad, 2.0f) + eps);
    float nvx = fl_div(d2x, scale);
    float nvy = fl_div(d2y, scale);
    float rb, rw, rs;
    fl_fire_ros(&rb, &rw, &rs, nvx, nvy, vx[f], vy[f], dzdx[f], dzdy[f],
                r0[f], bbb[f], phiwc[f], betafl[f], ischap[f], advection);
    float rr = rb + rw + slope_factor * rs;
    rr = fmaxf(rr, 0.0f);                           // fire_grows_only = 1
    ros[f] = rr;
    float te = -(rr * grad);
    float cb = 0.0f;
    if (grad > 0.0f) {
        cb = fl_div(rr * (fl_div(fabsf(d2x), dx) + fl_div(fabsf(d2y), dy)),
                    grad);
    }
    *cb_out = cb;
    float al = fabsf(l);
    bool inner = i > 1 + FL_BDY && i < nx - FL_BDY && j > 1 + FL_BDY
              && j < ny - FL_BDY;
    float var;
    if (al < thr_av && inner) {
        var = visc_bg;
    } else if (al >= thr_av && al < thr_av * (1.0f + visc_band) && inner) {
        var = fminf(visc_bg + fl_div((visc - visc_bg) * (al - thr_av),
                                     visc_band * thr_av), visc);
    } else {
        var = visc;
    }
    te = te + var * fabsf(rr) * ((drx - dlx) + (dry - dly));
    out[c] = lfn0[c] + coef * te;
}

extern "C" __global__ void fl_tend_stage(
    const float *__restrict__ lfn, const float *__restrict__ lfn0,
    float *__restrict__ out, float *__restrict__ ros,
    float *__restrict__ partial, const float *__restrict__ vx,
    const float *__restrict__ vy, const float *__restrict__ dzdx,
    const float *__restrict__ dzdy, const float *__restrict__ r0,
    const float *__restrict__ bbb, const float *__restrict__ phiwc,
    const float *__restrict__ betafl, const float *__restrict__ ischap,
    float dx, float dy, float coef,
    float thr_hl, float thr_av, float visc, float visc_bg, float visc_band,
    float slope_factor, int advection, int nx, int ny)
{
    __shared__ float part[FL_TPB];
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int t = threadIdx.y * blockDim.x + threadIdx.x;
    float cb = 0.0f;                                // tend_ls's tbound = 0
    if (i <= nx && j <= ny)
        fl_tend_node(lfn, lfn0, out, ros, &cb, vx, vy, dzdx, dzdy, r0, bbb,
                     phiwc, betafl, ischap, dx, dy, coef, thr_hl, thr_av, visc,
                     visc_bg, visc_band, slope_factor, advection, nx, ny, i,
                     j);
    part[t] = cb;
    __syncthreads();
    for (int s = FL_TPB / 2; s > 0; s >>= 1) {
        if (t < s) part[t] = fmaxf(part[t], part[t + s]);
        __syncthreads();
    }
    if (t == 0) partial[blockIdx.y * gridDim.x + blockIdx.x] = part[0];
}

// One block over a stage's per-block maxima: the CFL bound 1/(max + tol)
// into tb[stage]; the third stage also writes prop_ls_rk3's
// min(tbound, tbound2, tbound3) to tb[3].  MAX is exact in any order, so the
// tree is the reference's answer.
extern "C" __global__ void fl_bound_reduce(const float *__restrict__ bound,
                                           int n, float tol, float *tb,
                                           int stage)
{
    __shared__ float part[FL_TPB];
    float m = 0.0f;
    for (int k = threadIdx.x; k < n; k += blockDim.x) m = fmaxf(m, bound[k]);
    part[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            part[threadIdx.x] = fmaxf(part[threadIdx.x],
                                      part[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float t = fl_div(1.0f, part[0] + tol);
        tb[stage] = t;
        if (stage == 2) tb[3] = fminf(fminf(tb[0], tb[1]), t);
    }
}

extern "C" __global__ void fl_tign_update(const float *__restrict__ lfn_in,
                                          const float *__restrict__ lfn_out,
                                          float *__restrict__ tign, float ts,
                                          float dt, float time_now, int nx,
                                          int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int c = j * (nx + 2) + i;
    float li = lfn_in[c];
    float lo = lfn_out[c];
    if (!(lo > 0.0f) && li > 0.0f) tign[c] = ts + fl_div(dt * li, li - lo);
    if (lo > 0.0f) tign[c] = time_now;
}

// dx2 is dx**2.0, glibc's powf of the node spacing: the same word at every
// node, so the host evaluates it once (woof.core.noahmp_libm.powf).
extern "C" __global__ void fl_reinit_init(const float *__restrict__ lfn2,
                                          float *__restrict__ s0,
                                          float *__restrict__ s3, float dx2,
                                          int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int c = j * (nx + 2) + i;
    float l = lfn2[c];
    s0[c] = fl_div(l, fl_sqrt(gfk_pow(l, 2.0f) + dx2));
    s3[c] = l;
}

extern "C" __global__ void fl_reinit_stage(const float *__restrict__ s0,
                                           const float *ini,
                                           const float *curr, float *fin,
                                           float dx, float dy, float coef,
                                           float thr_hl, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int p = nx + 2;
    int c = j * p + i;
    float l = curr[c];
    float d2x, d2y;
    bool band = i < 1 + FL_BDY || i > nx - FL_BDY || j < 1 + FL_BDY
             || j > ny - FL_BDY;
    if (band || !(l < thr_hl)) {
        float dlx = fl_div(l - curr[c - 1], dx);
        float dly = fl_div(l - curr[c - p], dy);
        float drx = fl_div(curr[c + 1] - l, dx);
        float dry = fl_div(curr[c + p] - l, dy);
        d2x = fl_select_eno(dlx, drx);
        d2y = fl_select_eno(dly, dry);
    } else {
        float sg = s0[c];
        d2x = fl_select_4th(dx, l, curr[c - 1], curr[c - 2], curr[c + 1],
                            curr[c + 2]);
        d2y = fl_select_4th(dy, l, curr[c - p], curr[c - 2 * p],
                            curr[c + p], curr[c + 2 * p]);
        d2x = fl_select_weno5(dx, l, curr[c - 1], curr[c - 2], curr[c - 3],
                              curr[c + 1], curr[c + 2], curr[c + 3],
                              sg * d2x);
        d2y = fl_select_weno5(dy, l, curr[c - p], curr[c - 2 * p],
                              curr[c - 3 * p], curr[c + p], curr[c + 2 * p],
                              curr[c + 3 * p], sg * d2y);
    }
    float grad = fl_sqrt(d2x * d2x + d2y * d2y);
    float tend = s0[c] * (1.0f - grad);
    fin[c] = ini[c] + coef * tend;
}

// reinit_ls_rk3's lfn_out = min(lfn_s3, lfn_in), then fire_model pass 5's
// lfn = lfn_out.  Without reinitialization the caller passes lfn_out as s3
// and min_in = 0, which is the plain copy.
extern "C" __global__ void fl_reinit_final(const float *__restrict__ s3,
                                           float *lfn, int min_in, int nx,
                                           int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int c = j * (nx + 2) + i;
    float v = s3[c];
    if (min_in) v = fminf(v, lfn[c]);
    lfn[c] = v;
}

// ignite_fire with nearest inlined at one node of coordinates (ax, ay).
__device__ void fl_ignite_node(float *lfn, float *tign, int c, float ax,
                               float ay, float sx, float sy, float ex,
                               float ey, float st, float et, float cx2,
                               float cy2, float radius, float ros,
                               float start_ts, float end_ts)
{
    // nearest
    float mx = (sx + ex) * 0.5f;
    float my = (sy + ey) * 0.5f;
    float dam2 = (ax - mx) * (ax - mx) * cx2 + (ay - my) * (ay - my) * cy2;
    float des2 = (ex - sx) * (ex - sx) * cx2 + (ey - sy) * (ey - sy) * cy2;
    float dames = dam2 * des2;
    float am_es = (ax - mx) * (ex - sx) * cx2 + (ay - my) * (ey - sy) * cy2;
    float cos2;
    if (dames > 0.0f) cos2 = fl_div(am_es * am_es, dames);
    else cos2 = 0.0f;
    float dmc2 = dam2 * cos2;
    float mcrel;
    if (4.0f * dmc2 < des2) mcrel = copysignf(fl_sqrt(fl_div(4.0f * dmc2,
                                                             des2)), am_es);
    else if (am_es > 0.0f) mcrel = 1.0f;
    else mcrel = -1.0f;
    float cx = (ex + sx) * 0.5f + mcrel * (ex - sx) * 0.5f;
    float cy = (ey + sy) * 0.5f + mcrel * (ey - sy) * 0.5f;
    float d = fl_sqrt((ax - cx) * (ax - cx) * cx2 + (ay - cy) * (ay - cy) * cy2);
    float t = (et + st) * 0.5f + mcrel * (et - st) * 0.5f;
    // ignite_fire
    float lfn_new = d - fminf(radius, ros * (end_ts - t));
    float l = lfn[c];
    if (l > 0.0f && !(lfn_new > 0.0f)) {
        float ti = t + fl_div(d, ros);
        if (ti < start_ts || ti > end_ts) ti = fminf(fmaxf(ti, start_ts), end_ts);
        tign[c] = ti;
    }
    lfn[c] = fminf(l, lfn_new);
}

extern "C" __global__ void fl_ignite(float *lfn, float *tign, float sx,
                                     float sy, float ex, float ey, float st,
                                     float et, float cx2, float cy2,
                                     float radius, float ros, float start_ts,
                                     float end_ts, float fdx, float fdy,
                                     int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    // set_ideal_coord: (i - ifds + 0.5) * dxf
    float ax = ((float)(i - 1) + 0.5f) * fdx;
    float ay = ((float)(j - 1) + 0.5f) * fdy;
    fl_ignite_node(lfn, tign, j * (nx + 2) + i, ax, ay, sx, sy, ex, ey, st,
                   et, cx2, cy2, radius, ros, start_ts, end_ts);
}

// The same at given node coordinates (WRF's fxlong/fxlat for an ignition
// in longitude/latitude), (ny, nx) row-major.
extern "C" __global__ void fl_ignite_xy(float *lfn, float *tign, float sx,
                                        float sy, float ex, float ey,
                                        float st, float et, float cx2,
                                        float cy2, float radius, float ros,
                                        float start_ts, float end_ts,
                                        const float *coord_x,
                                        const float *coord_y, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int f = (j - 1) * nx + (i - 1);
    fl_ignite_node(lfn, tign, j * (nx + 2) + i, coord_x[f], coord_y[f], sx,
                   sy, ex, ey, st, et, cx2, cy2, radius, ros, start_ts,
                   end_ts);
}

// fuel_left_cell_1 for one subcell, corners (00, 01, 10, 11).
__device__ void fl_fuel_cell(float *left, float *area_out, float l00,
                             float l01, float l10, float l11, float g00,
                             float g01, float g10, float g11, float time_now,
                             float fuel_time)
{
    const float safe = 0x1p-126f;                   // tiny(aps)
    float t00 = g00 - time_now;
    if (l00 > 0.0f || t00 > 0.0f) t00 = 0.0f;
    float t01 = g01 - time_now;
    if (l01 > 0.0f || t01 > 0.0f) t01 = 0.0f;
    float t10 = g10 - time_now;
    if (l10 > 0.0f || t10 > 0.0f) t10 = 0.0f;
    float t11 = g11 - time_now;
    if (l11 > 0.0f || t11 > 0.0f) t11 = 0.0f;
    float ps = l00 + l01 + l10 + l11;
    float aps = fabsf(l00) + fabsf(l01) + fabsf(l10) + fabsf(l11);
    aps = fmaxf(aps, safe);
    float area = fl_div(-fl_div(ps, aps) + 1.0f, 2.0f);
    area = fmaxf(area, 0.0f);
    area = fminf(area, 1.0f);
    float ta = 0.25f * (t00 + t01 + t10 + t11);
    float out = 1.0f;
    if (area > 0.0f) out = area * gfk_exp(fl_div(ta, fuel_time)) + (1.0f - area);
    *left = out;
    *area_out = area;
}

// The bilinear corner interpolants of one subcell (fuel_left's lff/tif).
__device__ __forceinline__ float fl_bilin(float tx, float ty, float a00,
                                          float a01, float a10, float a11)
{
    return (1.0f - tx) * (1.0f - ty) * a00 + (1.0f - tx) * ty * a01
         + tx * (1.0f - ty) * a10 + tx * ty * a11;
}

extern "C" __global__ void fl_fuel_heat(
    const float *__restrict__ lfn, const float *__restrict__ tign,
    const float *__restrict__ fuel_time, float *__restrict__ fuel_frac,
    float *__restrict__ fire_area, float *__restrict__ burnt,
    const float *__restrict__ fgip, const float *__restrict__ fmc_g,
    float *__restrict__ grnhfx, float *__restrict__ grnqfx, float time_now,
    float dt, float cmbcnst, float xlv, int nx, int ny)
{
    int icl = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jcl = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (icl > nx || jcl > ny) return;
    int p = nx + 2;
    int f = (jcl - 1) * nx + (icl - 1);
    float ft = fuel_time[f];
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int isub = 1; isub <= 2; ++isub) {
        for (int jsub = 1; jsub <= 2; ++jsub) {
            int i2, j2;
            float tx, ty, txx, tyy;
            if (isub == 1 && jsub == 1) {
                i2 = icl - 1; j2 = jcl - 1; ty = 0.5f; tx = 0.5f; tyy = 1.0f;
                txx = 1.0f;
            } else if (isub == 2 && jsub == 1) {
                i2 = icl; j2 = jcl - 1; ty = 0.5f; tx = 0.0f; tyy = 1.0f;
                txx = 0.5f;
            } else if (isub == 1 && jsub == 2) {
                i2 = icl - 1; j2 = jcl; tx = 0.5f; ty = 0.0f; txx = 1.0f;
                tyy = 0.5f;
            } else {
                i2 = icl; j2 = jcl; tx = 0.0f; ty = 0.0f; txx = 0.5f;
                tyy = 0.5f;
            }
            int c00 = j2 * p + i2;
            float a00 = lfn[c00], a01 = lfn[c00 + p], a10 = lfn[c00 + 1],
                  a11 = lfn[c00 + p + 1];
            float g00 = tign[c00], g01 = tign[c00 + p], g10 = tign[c00 + 1],
                  g11 = tign[c00 + p + 1];
            float lij = fl_bilin(tx, ty, a00, a01, a10, a11);
            float li1j = fl_bilin(txx, ty, a00, a01, a10, a11);
            float lij1 = fl_bilin(tx, tyy, a00, a01, a10, a11);
            float li1j1 = fl_bilin(txx, tyy, a00, a01, a10, a11);
            float tij = fl_bilin(tx, ty, g00, g01, g10, g11);
            float ti1j = fl_bilin(txx, ty, g00, g01, g10, g11);
            float tij1 = fl_bilin(tx, tyy, g00, g01, g10, g11);
            float ti1j1 = fl_bilin(txx, tyy, g00, g01, g10, g11);
            float left, area;
            fl_fuel_cell(&left, &area, lij, lij1, li1j, li1j1, tij, tij1,
                         ti1j, ti1j1, time_now, ft);
            sum1 = sum1 + left;
            sum2 = sum2 + area;
        }
    }
    float end = fl_div(sum1, 4.0f);
    fire_area[f] = fl_div(sum2, 4.0f);
    float b = fuel_frac[f] - end;
    burnt[f] = b;
    fuel_frac[f] = end;
    // heat_fluxes
    float dmass = fgip[f] * b;
    float m = fmc_g[f];
    float bmst = fl_div(m, 1.0f + m);
    float rate = fl_div(dmass, dt);
    grnhfx[f] = rate * (1.0f - bmst) * cmbcnst;
    grnqfx[f] = (bmst + (1.0f - bmst) * 0.56f) * rate * xlv;
}

// sum_2d_cells over each atmospheric cell in WRF's (joff outer, ioff inner)
// order, then the driver's feedback * sum * (1/(ir*jr)).
extern "C" __global__ void fl_sum_cells(const float *__restrict__ fine,
                                        float *__restrict__ coarse, int ir,
                                        int jr, float feedback, float s,
                                        int use_feedback, int nx_atm,
                                        int ny_atm)
{
    int i1 = blockIdx.x * blockDim.x + threadIdx.x;
    int j1 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i1 >= nx_atm || j1 >= ny_atm) return;
    int nx = nx_atm * ir;
    int ibase = ir * i1;
    int jbase = jr * j1;
    float t = 0.0f;
    for (int joff = 0; joff < jr; ++joff)
        for (int ioff = 0; ioff < ir; ++ioff)
            t = t + fine[(jbase + joff) * nx + ibase + ioff];
    coarse[j1 * nx_atm + i1] = use_feedback ? feedback * t * s : t * s;
}

extern "C" __global__ void fl_accumulate(float *__restrict__ acc,
                                         const float *__restrict__ x, int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) acc[k] = acc[k] + x[k];
}

extern "C" __global__ void fl_scale(float *__restrict__ acc, float divisor,
                                    int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) acc[k] = fl_div(acc[k], divisor);
}
