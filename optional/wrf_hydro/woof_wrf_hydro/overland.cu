// WOOF-OPTIN: wrf-hydro
// Derived from WRF-Hydro (UCAR) as shipped in WRF 4.6.1 hydro/; see
// optional/wrf_hydro/LICENSE and NOTICE.md.  Not part of the default build or
// of any wheel.
//
// WRF-Hydro's overland routing, rt_option = 1:
// hydro/Routing/Noah_distr_routing_overland.F90 (OverlandRouting, OV_RTNG,
// ROUTE_OVERLAND1) and Noah_distr_routing.F90 (GETMAX8DIR, GET8DIR).  Planes
// are (jxrt, ixrt) C order, i fastest; sdir holds WRF's 1-based (i, j, k).
// ROUTE_OVERLAND1's REAL*8 scatter (DH_tmp, QBDRY_tmp) becomes a gather in
// WRF's j-outer, i-inner order; QBDRYT and the channel/lake accumulators are
// single-thread running sums in that order.  hh**(5./3.) is glibc's powf.

// OV_RTNG's channel and lake retention (:258-290), per cell.  dch/dlk are the
// amounts this cell adds to the two domain accumulators (0 when none).
extern "C" __global__
void ov_retention(const int* __restrict__ ch_netrt, const int* __restrict__ lake_mask,
                  float* __restrict__ retention, float* __restrict__ head,
                  float* __restrict__ to_channel, float* __restrict__ to_lake,
                  float* __restrict__ dch, float* __restrict__ dlk, int n)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float h = head[c];
    float a = 0.0f, b = 0.0f;
    if (ch_netrt[c] >= 0) {
        retention[c] = 5.0f;
        if (h > retention[c]) {
            a = __fsub_rn(h, retention[c]);
            to_channel[c] = __fadd_rn(to_channel[c], __fsub_rn(h, retention[c]));
            h = retention[c];
        }
    }
    if (lake_mask[c] > 0) {
        if (h > retention[c]) {
            b = __fsub_rn(h, retention[c]);
            to_lake[c] = __fadd_rn(to_lake[c], __fsub_rn(h, retention[c]));
            h = retention[c];
        }
    }
    head[c] = h;
    dch[c] = a;
    dlk[c] = b;
}

extern "C" __global__
void ov_retention_total(const int* __restrict__ ch_netrt, const int* __restrict__ lake_mask,
                        const float* __restrict__ dch, const float* __restrict__ dlk,
                        float* __restrict__ acc, int n)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float ach = acc[0], alk = acc[1];
    for (int c = 0; c < n; ++c) {
        if (dch[c] != 0.0f) ach = __fadd_rn(ach, dch[c]);
        if (dlk[c] != 0.0f) alk = __fadd_rn(alk, dlk[c]);
    }
    acc[0] = ach;
    acc[1] = alk;
}

// ROUTE_OVERLAND1's per-cell part (:458-575) with GETMAX8DIR/GET8DIR.
extern "C" __global__
void ov_route_cell(float dt, const float* __restrict__ gsize, const float* __restrict__ h,
                   const float* __restrict__ retent, const float* __restrict__ rough,
                   const float* __restrict__ so8rt, const int* __restrict__ so8rt_d,
                   float* __restrict__ qx, float* __restrict__ qy,
                   float* __restrict__ adjust, float* __restrict__ qq,
                   int* __restrict__ target, int* __restrict__ boundary, int xx, int yy)
{
    int n = xx * yy;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int i = c % xx + 1, j = c / xx + 1;
    target[c] = -1;
    boundary[c] = 0;
    adjust[c] = 0.0f;
    qq[c] = 0.0f;
    if (!(i >= 2 && i <= xx - 1 && j >= 2 && j <= yy - 1)) return;
    float hc = h[c];
    if (!(hc > retent[c])) return;
    int ixx0 = so8rt_d[c], jyy0 = so8rt_d[n + c], index = so8rt_d[2 * n + c];
    float tmp_gsize = __fdiv_rn(1.0f, gsize[(size_t)(index - 1) * n + c]);
    float h0 = h[(jyy0 - 1) * xx + (ixx0 - 1)];
    float sfx = __fsub_rn(so8rt[(size_t)(index - 1) * n + c],
                          __fmul_rn(__fmul_rn(__fsub_rn(h0, hc), 0.001f), tmp_gsize));
    float hmax = __fmul_rn(hc, 0.001f);
    if (sfx < 1E-20f) {
        // GETMAX8DIR: IXX0 = -1, max = 0, then GET8DIR in N, NE, E, SE, S,
        // SW, W, NW order; a neighbour counts only with a positive slope.
        const int ni[8] = {i, i + 1, i + 1, i + 1, i, i - 1, i - 1, i - 1};
        const int nj[8] = {j + 1, j + 1, j, j - 1, j - 1, j - 1, j, j + 1};
        ixx0 = -1;
        float mx = 0.0f;
        for (int k = 0; k < 8; ++k) {
            float h8 = h[(nj[k] - 1) * xx + (ni[k] - 1)];
            float s8 = __fsub_rn(so8rt[(size_t)k * n + c],
                                 __fdiv_rn(__fmul_rn(__fsub_rn(h8, hc), 0.001f),
                                           gsize[(size_t)k * n + c]));
            if (s8 <= 0.0f) continue;
            if (mx < s8) {
                ixx0 = ni[k];
                jyy0 = nj[k];
                mx = s8;
            }
        }
        sfx = mx;
    }
    if (ixx0 <= 0) return;
    if (sfx < 1E-20f) sfx = 1E-20f;
    float alfax = __fdiv_rn(__fsqrt_rn(sfx), rough[c]);
    float hh = __fmul_rn(__fsub_rn(hc, retent[c]), 0.001f);
    float hh53 = gfk_pow(hh, 5.0f / 3.0f);
    float qqsfc = __fmul_rn(__fmul_rn(__fmul_rn(alfax, hh53), dt), tmp_gsize);
    float cap = __fmul_rn(__fmul_rn(hmax, dt), tmp_gsize);
    if (qqsfc >= cap) qqsfc = cap;
    float fy = __fsub_rn(1.0f, __fmul_rn(0.5f, (float)abs(j - jyy0)));
    if (ixx0 > i) qx[c] = __fadd_rn(qx[c], __fmul_rn(qqsfc, fy));
    else if (ixx0 < i) qx[c] = __fsub_rn(qx[c], __fmul_rn(__fmul_rn(1.0f, qqsfc), fy));
    else qx[c] = __fadd_rn(qx[c], 0.0f);
    float fx = __fsub_rn(1.0f, __fmul_rn(0.5f, (float)abs(i - ixx0)));
    if (jyy0 > j) qy[c] = __fadd_rn(qy[c], __fmul_rn(qqsfc, fx));
    else if (jyy0 < j) qy[c] = __fsub_rn(qy[c], __fmul_rn(__fmul_rn(1.0f, qqsfc), fx));
    else qy[c] = __fadd_rn(qy[c], 0.0f);
    float adj = __fmul_rn(qqsfc, 1000.0f);
    if (__fsub_rn(hc, adj) < 0.0f) adj = hc;
    adjust[c] = adj;
    qq[c] = qqsfc;
    target[c] = (jyy0 - 1) * xx + (ixx0 - 1);
    boundary[c] = (ixx0 == xx) || (ixx0 == 1) || (jyy0 == 1) || (jyy0 == yy);
}

// The rest of ROUTE_OVERLAND1 (:590-604): DH = -own + DH_tmp, H = H + DH,
// QBDRY = QBDRY + QBDRY_tmp, each cell gathering what its neighbours sent.
extern "C" __global__
void ov_gather(const float* __restrict__ adjust, const float* __restrict__ qq,
               const int* __restrict__ target, const int* __restrict__ boundary,
               float* __restrict__ h, float* __restrict__ qbdry, int xx, int yy)
{
    int n = xx * yy;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int ti = c % xx + 1, tj = c / xx + 1;
    double dh = 0.0;
    if (target[c] >= 0) dh = __dsub_rn(dh, (double)adjust[c]);
    double dht = 0.0, qbt = 0.0;
    for (int sj = tj - 1; sj <= tj + 1; ++sj)
        for (int si = ti - 1; si <= ti + 1; ++si) {
            if (si < 2 || si > xx - 1 || sj < 2 || sj > yy - 1) continue;
            int s = (sj - 1) * xx + (si - 1);
            if (s == c || target[s] != c) continue;
            dht = __dadd_rn(dht, (double)adjust[s]);
            if (boundary[s]) {
                qbt = __dsub_rn(qbt, (double)__fmul_rn(qq[s], 1000.0f));
                dht = __dsub_rn(dht, (double)adjust[s]);
            }
        }
    qbdry[c] = __double2float_rn(__dadd_rn((double)qbdry[c], qbt));
    dh = __dadd_rn(dh, dht);
    h[c] = __double2float_rn(__dadd_rn((double)h[c], dh));
}

extern "C" __global__
void ov_total(const float* __restrict__ qq, const int* __restrict__ target,
              const int* __restrict__ boundary, float* __restrict__ qbdryt, int xx, int yy)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float t = qbdryt[0];
    for (int j = 2; j <= yy - 1; ++j)
        for (int i = 2; i <= xx - 1; ++i) {
            int s = (j - 1) * xx + (i - 1);
            if (target[s] >= 0 && boundary[s]) t = __fsub_rn(t, qq[s]);
        }
    qbdryt[0] = t;
}

// The domain-edge drain (:606-656): head above retention on an edge cell
// leaves through the boundary.
extern "C" __global__
void ov_edge(float* __restrict__ h, float* __restrict__ qbdry,
             const float* __restrict__ retent, int xx, int yy)
{
    int n = xx * yy;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int i = c % xx + 1, j = c / xx + 1;
    float e = 0.0f;
    if ((i == xx) || (i == 1) || (j == 1) || (j == yy))
        if (h[c] > retent[c]) e = __fsub_rn(h[c], retent[c]);
    qbdry[c] = __fsub_rn(qbdry[c], e);
    h[c] = __fsub_rn(h[c], e);
}

// OverlandRouting's sfcrt_flag (:96-110): any cell's infiltration excess
// above its retention depth.
extern "C" __global__
void ov_any_above(const float* __restrict__ infxs, const float* __restrict__ retent,
                  int* __restrict__ flag, int n)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    if (infxs[c] > retent[c]) atomicOr(flag, 1);
}
