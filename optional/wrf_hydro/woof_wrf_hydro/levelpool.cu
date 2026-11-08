// WOOF-OPTIN: wrf-hydro
// Derived from WRF-Hydro as distributed with WRF 4.6.1 (UCAR licence); see
// optional/wrf_hydro/LICENSE and NOTICE.md.  Not part of the default build or
// of any wheel.
//
// WRF-Hydro's level-pool reservoir, LEVELPOOL_PHYSICS with LAKE_OPT = 2
// (hydro/Routing/Reservoirs/Level_Pool/module_levelpool.F90 :132-249): Chow's
// level pool stepped by a third-order Runge-Kutta in the water elevation, the
// outflow through an orifice, over a weir and over the dam.  float32 as
// WRF-Hydro, the Fortran expressions in Fortran order, powf as glibc's
// (gfk_pow, preceding this file).

__device__ static float lp_discharge(float Hx, float dh, float H, float maxh, float oe,
                                     float wc, float wl, float dl, float oc, float oa)
{
    // the orifice and weir terms are evaluated whether or not they are used,
    // as the Fortran does; Hx is the elevation of the RK stage
    float tmp1 = oc * oa * sqrtf(2.0f * 9.81f * (Hx - oe));
    float tmp2 = wc * wl * (gfk_pow(dh, 1.5f));
    if (H > maxh)
        return tmp1 + tmp2 + (wc * (wl * dl) * gfk_pow(H - maxh, 1.5f));
    else if (dh > 0.0f)
        return tmp1 + tmp2;
    else if (Hx > oe)
        return oc * oa * sqrtf(2.0f * 9.81f * (Hx - oe));
    return 0.0f;
}

__device__ static void levelpool_physics(float qi0, float qi1, float& qo1, float ql, float dt,
                                         float& H, float ar, float we, float maxh, float wc,
                                         float wl, float dl, float oe, float oc, float oa)
{
    float It = qi0;
    float Itdt_3 = qi0 + ((qi1 + ql - qi0) * 0.33f);
    float Itdt_2_3 = qi0 + ((qi1 + ql - qi0) * 0.67f);
    float maxWeirDepth = maxh - we;
    float sap = ar * 1.0E6f;
    float dh = H - we;
    if (dh > maxWeirDepth) dh = maxWeirDepth;
    float discharge = lp_discharge(H, dh, H, maxh, oe, wc, wl, dl, oc, oa);
    float dh1 = sap > 0.0f ? ((It - discharge) / sap) * dt : 0.0f;
    float H1 = H + dh1 / 3.0f;
    dh = H1 - we;
    if (dh > maxWeirDepth) dh = maxWeirDepth;
    discharge = lp_discharge(H1, dh, H, maxh, oe, wc, wl, dl, oc, oa);
    float dh2 = sap > 0.0f ? ((Itdt_3 - discharge) / sap) * dt : 0.0f;
    float H2 = H + (0.667f * dh2);
    dh = H2 - we;
    if (dh > maxWeirDepth) dh = maxWeirDepth;
    discharge = lp_discharge(H2, dh, H, maxh, oe, wc, wl, dl, oc, oa);
    float dh3 = sap > 0.0f ? ((Itdt_2_3 - discharge) / sap) * dt : 0.0f;
    dh = (dh1 / 4.0f) + (0.75f * dh3);
    H = H + dh;
    dh = H - we;
    if (dh > maxWeirDepth) dh = maxWeirDepth;
    qo1 = lp_discharge(H, dh, H, maxh, oe, wc, wl, dl, oc, oa);
}

// run_levelpool_reservoir (:88-131) on independent reservoirs, one thread each
extern "C" __global__ void wrfhydro_levelpool(
    const float* __restrict__ qi0, const float* __restrict__ qi1, const float* __restrict__ ql,
    float* __restrict__ h, float* __restrict__ qo, const float* __restrict__ props,
    float dt, int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const float* p = props + (size_t)k * 9;   // area, we, wc, wl, dl, oe, oc, oa, maxh
    float H = h[k], q;
    levelpool_physics(qi0[k], qi1[k], q, ql[k], dt, H, p[0], p[1], p[8], p[2], p[3], p[4],
                      p[5], p[6], p[7]);
    h[k] = H;
    qo[k] = q;
}
