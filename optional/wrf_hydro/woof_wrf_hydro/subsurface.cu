// WOOF-OPTIN: wrf-hydro
// Derived from WRF-Hydro (UCAR) as shipped in WRF 4.7.1 hydro/; see
// optional/wrf_hydro/LICENSE and NOTICE.md.  Not part of the default build or
// of any wheel.
//
// WRF-Hydro's subsurface routing, hydro/Routing/Noah_distr_routing_subsurface.F90
// (subsurfaceRouting -> FINDZWAT, ROUTE_SUBSURFACE1, SUBSFC_RTNG) and
// Noah_distr_routing.F90 (GETSUB8, GETSUB8DIR), on the routing grid.  Planes
// are (jxrt, ixrt) C order, i.e. WRF's (IXRT, JXRT) memory order with i
// fastest; soil fields are (nsoil, jxrt, ixrt).  Indices stored in sdir are
// WRF's 1-based (i, j, direction).  Every operation is in WRF's order with
// round-to-nearest intrinsics; (1 - z/soldep)**nexp is glibc's powf
// (gfk_pow), as the -O0 reference calls powf.
//
// ROUTE_SUBSURFACE1 scatters each cell's outflow into a REAL*8 array at its
// steepest neighbour.  Here each cell gathers what its eight neighbours send
// it, visiting them in the order WRF's j-outer, i-inner loop reaches them, so
// every double sum is formed in WRF's order.

#define HYD_ERR_DIST    1   // dist(i,j,index) <= 0
#define HYD_ERR_SOLDEP  2   // soldep == 0
#define HYD_ERR_HSUB    4   // hh < 0
#define HYD_ERR_QQSUB   8   // qqsub > 0
#define HYD_ERR_SMCMAX 16   // SMC above SMCMAX in SUBSFC_RTNG

// FINDZWAT (Noah_distr_routing_subsurface.F90:457-543), one thread per cell.
extern "C" __global__
void hyd_findzwat(const float* __restrict__ smc, const float* __restrict__ smcref,
                  const float* __restrict__ smcwlt, const float* __restrict__ zsoil,
                  const float* __restrict__ sldpth, float* __restrict__ zwat,
                  float* __restrict__ cwat, int* __restrict__ satlyr,
                  int nsoil, int n)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int sat = 0;
    float cw = 0.0f;
    for (int kk = nsoil; kk >= 1; --kk) {
        size_t o = (size_t)(kk - 1) * n + c;
        if ((smc[o] >= smcref[o]) && (smcref[o] > smcwlt[o])) {
            if ((sat == kk + 1) || (kk == nsoil)) sat = kk;
        }
    }
    float z;
    if (sat != 0) {
        if (sat != 1) z = -zsoil[sat - 2];
        else z = 0.0f;
        for (int kk = sat; kk <= nsoil; ++kk) {
            size_t o = (size_t)(kk - 1) * n + c;
            cw = __fadd_rn(cw, __fmul_rn(__fsub_rn(smc[o], smcref[o]), sldpth[kk - 1]));
        }
    } else {
        z = -zsoil[nsoil - 1];
        sat = nsoil + 1;
    }
    zwat[c] = z;
    cwat[c] = cw;
    satlyr[c] = sat;
}

// ROUTE_SUBSURFACE1's per-cell part (:604-670) with GETSUB8: the outflow
// qqsub to the steepest total-head neighbour, the cell's own qsub update,
// and the target.  target = -1 where WRF computes no flux.
extern "C" __global__
void hyd_route1_cell(const float* __restrict__ dist, const float* __restrict__ z,
                     const float* __restrict__ latksat, const float* __restrict__ nexp,
                     const float* __restrict__ soldep, const float* __restrict__ so8rt,
                     const int* __restrict__ so8rt_d, const float* __restrict__ cwat,
                     float subdt, float* __restrict__ qsub, float* __restrict__ qq,
                     int* __restrict__ target, int* __restrict__ boundary,
                     int* __restrict__ err, int xx, int yy)
{
    int n = xx * yy;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int i = c % xx + 1, j = c / xx + 1;
    qq[c] = 0.0f;
    target[c] = -1;
    boundary[c] = 0;
    if (!(i >= 2 && i <= xx - 1 && j >= 2 && j <= yy - 1)) return;
    int ixx0 = so8rt_d[c], jyy0 = so8rt_d[n + c], index = so8rt_d[2 * n + c];
    // GETSUB8 (Noah_distr_routing.F90:155-189) and GETSUB8DIR (:191-229)
    const int ni[8] = {i, i + 1, i + 1, i + 1, i, i - 1, i - 1, i - 1};
    const int nj[8] = {j + 1, j + 1, j, j - 1, j - 1, j - 1, j, j + 1};
    float beta = -1.0f;
    float zself = z[c];
    for (int k = 0; k < 8; ++k) {
        int nc = (nj[k] - 1) * xx + (ni[k] - 1);
        float dzdx = __fdiv_rn(__fsub_rn(zself, z[nc]), dist[(size_t)k * n + c]);
        float b = __fsub_rn(so8rt[(size_t)k * n + c], dzdx);
        if (beta < b) {
            ixx0 = ni[k];
            jyy0 = nj[k];
            beta = b;
            index = k + 1;
        }
    }
    float d_index = dist[(size_t)(index - 1) * n + c];
    if (d_index <= 0.0f) atomicOr(err, HYD_ERR_DIST);
    if (soldep[c] == 0.0f) atomicOr(err, HYD_ERR_SOLDEP);
    if (beta > 0.0f) {
        if (beta < 1e-20f) beta = 1e-20f;
        float hh = gfk_pow(__fsub_rn(1.0f, __fdiv_rn(z[c], soldep[c])), nexp[c]);
        float ksat = latksat[c];
        if (hh < 0.0f) atomicOr(err, HYD_ERR_HSUB);
        float gamma = __fmul_rn(__fmul_rn(-1.0f, __fdiv_rn(__fmul_rn(__fmul_rn(
            d_index, ksat), soldep[c]), nexp[c])), beta);
        float qqsub = __fmul_rn(gamma, hh);
        float dist9 = dist[(size_t)8 * n + c];
        float water = __fmul_rn(__fdiv_rn(fabsf(qqsub), dist9), subdt);
        if ((qqsub <= 0.0f) && (cwat[c] < water))
            qqsub = __fdiv_rn(__fmul_rn(__fmul_rn(-1.0f, cwat[c]), dist9), subdt);
        qsub[c] = __fadd_rn(qsub[c], qqsub);
        if (qqsub > 0.0f) atomicOr(err, HYD_ERR_QQSUB);
        qq[c] = qqsub;
        target[c] = (jyy0 - 1) * xx + (ixx0 - 1);
        boundary[c] = (ixx0 == 1) || (ixx0 == xx) || (jyy0 == 1) || (jyy0 == yy);
    }
}

// The rest of ROUTE_SUBSURFACE1 (:633-688): each cell gathers the REAL*8
// qsub_tmp and QSUBDRY_tmp its neighbours scattered at it, in WRF's loop
// order, then qsub = qsub + qsub_tmp and QSUBDRY = QSUBDRY + QSUBDRY_tmp.
extern "C" __global__
void hyd_route1_gather(const float* __restrict__ qq, const int* __restrict__ target,
                       const int* __restrict__ boundary, float* __restrict__ qsub,
                       float* __restrict__ qsubdry, int xx, int yy)
{
    int n = xx * yy;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int ti = c % xx + 1, tj = c / xx + 1;
    double acc = 0.0, accb = 0.0;
    for (int sj = tj - 1; sj <= tj + 1; ++sj) {
        for (int si = ti - 1; si <= ti + 1; ++si) {
            if (si < 2 || si > xx - 1 || sj < 2 || sj > yy - 1) continue;
            int s = (sj - 1) * xx + (si - 1);
            if (s == c || target[s] != c) continue;
            double q = (double)qq[s];
            acc = __dsub_rn(acc, q);
            if (boundary[s]) {
                acc = __dadd_rn(acc, q);
                accb = __dadd_rn(accb, q);
            }
        }
    }
    qsub[c] = __double2float_rn(__dadd_rn((double)qsub[c], acc));
    qsubdry[c] = __double2float_rn(__dadd_rn((double)qsubdry[c], accb));
}

// QSUBDRYT = QSUBDRYT + qqsub over boundary-bound outflow, a float running
// sum in WRF's loop order: one thread.
extern "C" __global__
void hyd_route1_total(const float* __restrict__ qq, const int* __restrict__ target,
                      const int* __restrict__ boundary, float* __restrict__ total,
                      int xx, int yy)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float t = total[0];
    for (int j = 2; j <= yy - 1; ++j)
        for (int i = 2; i <= xx - 1; ++i) {
            int s = (j - 1) * xx + (i - 1);
            if (target[s] >= 0 && boundary[s]) t = __fadd_rn(t, qq[s]);
        }
    total[0] = t;
}

// SUBSFC_RTNG's soil-moisture accounting (:246-382), one thread per cell.
extern "C" __global__
void hyd_subsfc_update(const float* __restrict__ qsub, const float* __restrict__ dist,
                       const int* __restrict__ satlyr, const float* __restrict__ sldpth,
                       const float* __restrict__ smcmax, const float* __restrict__ smcref,
                       float* __restrict__ smc, float* __restrict__ infxs,
                       float subdt, int* __restrict__ err, int nsoil, int n)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float subflo = __fmul_rn(__fdiv_rn(qsub[c], dist[(size_t)8 * n + c]), subdt);
    if (subflo > 0.0f) {
        int kk;
        for (kk = nsoil; kk >= 1; --kk) {
            size_t o = (size_t)(kk - 1) * n + c;
            if (smc[o] >= smcmax[o]) {
                if (smc[o] > smcmax[o]) atomicOr(err, HYD_ERR_SMCMAX);
            } else {
                float watavail = __fmul_rn(__fsub_rn(smcmax[o], smc[o]), sldpth[kk - 1]);
                if (watavail >= subflo) {
                    smc[o] = __fadd_rn(smc[o], __fdiv_rn(subflo, sldpth[kk - 1]));
                    subflo = 0.0f;
                } else {
                    subflo = __fsub_rn(subflo, watavail);
                    smc[o] = smcmax[o];
                }
            }
            if (subflo == 0.0f) break;
        }
        if (kk == 0 && subflo > 0.0f) {
            infxs[c] = __fadd_rn(infxs[c], __fmul_rn(subflo, 1000.0f));
            subflo = 0.0f;
        }
    } else if (subflo < 0.0f) {
        for (int kk = satlyr[c]; kk <= nsoil; ++kk) {
            size_t o = (size_t)(kk - 1) * n + c;
            float watavail = __fmul_rn(__fsub_rn(smc[o], smcref[o]), sldpth[kk - 1]);
            if (watavail >= fabsf(subflo)) {
                smc[o] = __fadd_rn(smc[o], __fdiv_rn(subflo, sldpth[kk - 1]));
                subflo = 0.0f;
            } else {
                smc[o] = smcref[o];
                subflo = __fadd_rn(subflo, watavail);
            }
            if (subflo == 0.0f) break;
        }
    }
}
